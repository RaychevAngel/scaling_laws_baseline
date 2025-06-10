#!/usr/bin/env python3
import os
import tempfile
import shutil
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments
from utils.callbacks import LossPlotCallback

# Set tokenizers parallelism to avoid fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
IGNORE_VALUE = -1.0          # sentinel used to mark "not-a-label"

# ───────────────────────── Data collator wrapper ──────────────────── #
class ValueDataCollator:
    def __init__(self, tokenizer, pad_to_multiple_of):
        self._base = DataCollatorWithPadding(
            tokenizer,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )

    def __call__(self, features):
        raw_labels = [feat.pop("labels") for feat in features]
        
        batch = self._base(features)
        max_len = batch["input_ids"].size(1)
        device  = batch["input_ids"].device

        labels_padded = []
        for lab in raw_labels:
            if isinstance(lab, torch.Tensor):
                t = lab.clone().detach().to(device=device, dtype=torch.float32)
            else:
                t = torch.as_tensor(lab, dtype=torch.float32, device=device)

            pad_amount = max_len - t.size(0)
            if pad_amount > 0:
                t = F.pad(t, (0, pad_amount), value=IGNORE_VALUE)
            labels_padded.append(t)

        batch["labels"] = torch.stack(labels_padded)
        return batch

# ───────────────────────── Model wrapper ──────────────────────────── #
class ValueModel(torch.nn.Module):
    def __init__(self, model_name: str, revision: str | None, newline_token_ids: list[int], value_token_id: int):
        super().__init__()

        self.register_buffer(
            "newline_token_ids",
            torch.tensor(newline_token_ids, dtype=torch.long),
            persistent=False
        )
        self.value_token_id = value_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=torch.bfloat16,
            use_cache=False,
            device_map=None,
            ignore_mismatched_sizes=True
        )
            
        self.config = self.model.config

    @torch.autocast("cuda", dtype=torch.bfloat16)
    def forward(self, input_ids, attention_mask=None, labels=None):
        # logits of the full vocab
        logits = self.model(input_ids=input_ids,  attention_mask=attention_mask).logits # [B, L, V]
        logits_val = logits[..., self.value_token_id]        # [B, L]
        newline_mask = torch.isin(input_ids, self.newline_token_ids)  # [B, L]

        label_mask = labels.ne(IGNORE_VALUE)

        if torch.isnan(labels[label_mask]).any() or (newline_mask ^ label_mask).any():
            raise ValueError("Mismatch between newline positions and provided labels")

        logits_sel = logits_val[label_mask]                   # [N,]
        labels_sel = labels[label_mask]                       # [N,]

        loss = F.binary_cross_entropy_with_logits(logits_sel, labels_sel, reduction="mean")
        return {"loss": loss, "logits": logits_sel}

# ───────────────────────── Trainer wrapper ────────────────────────── #
class ValueTrainer:
    """High-level orchestrator: dataset ↔ tokenizer ↔ Trainer."""

    def __init__(self, config: dict):
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model_name"],
            revision=None if config["revision"] == "None" else config["revision"],
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.newline_token_ids = [
            token_id for token, token_id in self.tokenizer.get_vocab().items()
            if "\n" in self.tokenizer.decode([token_id])
        ]
        self.value_token_id = self.tokenizer.encode("1", add_special_tokens=False)[0]

        self.model = ValueModel(
            model_name=config["model_name"],
            revision=config["revision"] if config["revision"] != "None" else None,
            newline_token_ids=self.newline_token_ids,
            value_token_id=self.value_token_id
        )

        self.data_collator = ValueDataCollator(
            self.tokenizer,
            pad_to_multiple_of=8
        )
        self.temp_dir = tempfile.mkdtemp()
        self.final_loss = None

    # ───────────────────────── Tokenizer ──────────────────────────── #
    def _tokenize_dataset(self, dataset):
        newline_ids = torch.tensor(self.newline_token_ids)

        def encode(batch):
            enc = self.tokenizer(batch["text"], truncation=True)
            value_targets = []                        # one tensor per example

            for ids, label_list in zip(enc["input_ids"], batch["labels"]):
                ids_t = torch.tensor(ids)
                nl_pos = torch.isin(ids_t, newline_ids).nonzero().squeeze(1)

                # ---- sanity check ------------------------------------------------
                assert len(nl_pos) == len(label_list), (
                    f"#labels ({len(label_list)}) ≠ #newlines ({len(nl_pos)}) "
                    f"in text: {self.tokenizer.decode(ids)}"
                )

                tgt = torch.full((len(ids),), IGNORE_VALUE, dtype=torch.float32)
                tgt[nl_pos] = torch.tensor(label_list, dtype=torch.float32)
                value_targets.append(tgt)

            enc["labels"] = value_targets
            return enc

        dataset = dataset.map(encode, batched=True, remove_columns=["text"])
        return dataset

    def _create_training_args(self):
        return TrainingArguments(
            output_dir=self.temp_dir,
            
            per_device_train_batch_size=int(self.config["per_device_train_batch_size"]),
            gradient_accumulation_steps=int(self.config["gradient_accumulation_steps"]),
            num_train_epochs=int(self.config["num_train_epochs"]),
            learning_rate=float(self.config["learning_rate"]),
            lr_scheduler_type=self.config["lr_scheduler_type"],
            
            logging_steps=int(self.config["logging_steps"]),
            logging_first_step=True,
            
            save_strategy="no",
            
            bf16=True,
            dataloader_num_workers=64,
            dataloader_drop_last=True,
            save_safetensors=False,

            push_to_hub=True,
            hub_model_id=self.config["hub_model_id"],
            hub_strategy="end",
            hub_private_repo=False,
            disable_tqdm=True
        )

    def train(self):
        raw_dataset = load_from_disk(self.config["dataset_file"]).shuffle(seed=42)
        dataset = self._tokenize_dataset(raw_dataset)

        callback = LossPlotCallback(self.config["plot_path"], "Value")
        
        trainer = Trainer(
            model=self.model,
            args=self._create_training_args(),
            train_dataset=dataset,
            data_collator=self.data_collator,
            processing_class=self.tokenizer,
            callbacks=[callback],
        )

        trainer.args.train_dataset_size = len(dataset)
        effective_batch = trainer.args.per_device_train_batch_size * max(1, torch.cuda.device_count()) * trainer.args.gradient_accumulation_steps

        print(f"Per device train batch size: {trainer.args.per_device_train_batch_size}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Gradient accumulation steps: {trainer.args.gradient_accumulation_steps}")
        print(f"Effective batch size: {effective_batch}")
        print(f"Training dataset size: {trainer.args.train_dataset_size}")
        print(f"Steps per epoch: {trainer.args.train_dataset_size // effective_batch}")

        try:
            trainer.train()
            print("Training completed!")
        except KeyboardInterrupt:
            print("Training interrupted.")
            
        self.final_loss = callback.train_losses[-1] if callback.train_losses else None
        
        self.model.config.model_card = f"""
Final Loss: {str(self.final_loss)}
Batch Size: {effective_batch}
Learning Rate: {self.config['learning_rate']}
Dataset Size: {trainer.args.train_dataset_size}
"""

        # Push the final model
        print(f"Pushing model to Hugging Face Hub: {self.config['hub_model_id']}")
        try:
            self.model.model.push_to_hub(self.config["hub_model_id"], push_functional=True)
            self.tokenizer.push_to_hub(self.config["hub_model_id"])
            print(f"Model and tokenizer successfully pushed to: https://huggingface.co/{self.config['hub_model_id']}")
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Error pushing to hub: {e}")


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

# ───────────────────────── Model wrapper ──────────────────────────── #
class ValueModel(torch.nn.Module):
    def __init__(self, model_name: str, revision: str | None, newline_token_ids: list[int], value_token_id: int):
        super().__init__()
        self.newline_token_ids = newline_token_ids
        self.value_token_id = value_token_id
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=torch.bfloat16,
            use_cache=False,
            device_map="auto",
            ignore_mismatched_sizes=True,
        )
        # Add config attribute to make push_to_hub work
        self.config = self.model.config

    def forward(self, input_ids, attention_mask=None, labels=None):
        logits_full = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        logits_value = logits_full[..., self.value_token_id]  # [B, L]
        newline_mask = torch.zeros_like(input_ids[:, :-1], dtype=torch.bool)
        for token in self.newline_token_ids:
            newline_mask |= (input_ids[:, :-1] == token)
        logits_sel = logits_value[:, :-1][newline_mask]
        labels_sel = labels.unsqueeze(1).expand_as(newline_mask)[newline_mask]
        loss = F.binary_cross_entropy_with_logits(logits_sel, labels_sel, reduction="mean")
        return {"loss": loss, "logits": logits_sel}

# ───────────────────────── Trainer wrapper ────────────────────────── #
class ValueTrainer:
    """High‑level orchestrator: dataset ↔ tokenizer ↔ Trainer."""

    def __init__(self, config: dict):
        # Apply speed optimizations
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True  
        
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model_name"],
            revision=(None if config["revision"] == "None" else config["revision"]),
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token # required for padding

        # IDs needed for loss masking
        # Find all tokens containing newlines
        self.newline_token_ids = []
        for token, token_id in self.tokenizer.get_vocab().items():
            if '\n' in self.tokenizer.decode([token_id]):
                self.newline_token_ids.append(token_id)
        self.value_token_id = self.tokenizer.encode("1", add_special_tokens=False)[0]

        self.model = ValueModel(
            model_name=config["model_name"],
            revision=config["revision"] if config["revision"] != "None" else None,
            newline_token_ids=self.newline_token_ids,
            value_token_id=self.value_token_id
        )

        self.data_collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)
        self.temp_dir = tempfile.mkdtemp()
        self.final_loss = None

    def _tokenize_dataset(self, dataset):
        def encode(batch):
            encoded = self.tokenizer(batch["text"], truncation=True)
            # Ensure labels are 0/1 scalars per example (flatten nested lists if any)
            encoded["labels"] = [ (l[0] if isinstance(l, list) else l) for l in batch["labels"] ]
            return encoded
        dataset = dataset.map(encode, batched=True, remove_columns=["text"])
        dataset.set_format("torch")
        return dataset

    def _create_training_args(self):
        return TrainingArguments(
            output_dir=self.temp_dir,

            per_device_train_batch_size=int(self.config["per_device_train_batch_size"]),
            gradient_accumulation_steps=int(self.config["gradient_accumulation_steps"]),
            num_train_epochs=1,
            learning_rate=float(self.config["learning_rate"]),
            lr_scheduler_type=self.config["lr_scheduler_type"],

            logging_steps=int(self.config["logging_steps"]),
            logging_first_step=True,
            
            # Simple save steps
            save_strategy="no",

            bf16=True,
            dataloader_num_workers=64,
            save_safetensors=False,

            push_to_hub=True,
            hub_model_id=self.config["hub_model_id"],
            hub_strategy="end",
            hub_private_repo=False,
            disable_tqdm=True,
        )

    def train(self):
        # Load dataset 
        raw_dataset = load_from_disk(self.config["dataset_file"]).shuffle(seed=42)
        dataset = self._tokenize_dataset(raw_dataset)

        callback = LossPlotCallback(self.config["plot_path"], "Value")
        
        trainer = Trainer(
            model=self.model,
            args=self._create_training_args(),
            train_dataset=dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
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


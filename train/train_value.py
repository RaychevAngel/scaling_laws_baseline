#!/usr/bin/env python3
import os
import tempfile
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

# ───────────────────────── Callback utilities ──────────────────────────── #
class LossScalingCallback(TrainerCallback):
    """Scale the reported training loss when using gradient accumulation."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            logs["loss"] /= args.gradient_accumulation_steps


class EpochProgressBar(TrainerCallback):
    """Simple callback to show current epoch."""

    def __init__(self):
        self.current_epoch = 0

    def on_train_begin(self, args, state, control, **kwargs):
        print(f"Starting training, epoch 1")

    def on_step_end(self, args, state, control, **kwargs):
        epoch_int = int(state.epoch)
        if epoch_int != self.current_epoch:
            print(f"Epoch {epoch_int + 1}")
            self.current_epoch = epoch_int

    def on_train_end(self, *_, **__):
        print("Training completed")


class LossPlotCallback(TrainerCallback):
    """Interactive matplotlib plot of train/eval loss curves."""

    def __init__(self, plot_path: str):
        self.plot_path = Path(plot_path)
        self.train_losses, self.eval_losses = [], []
        self.train_steps, self.eval_steps = [], []
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        plt.ion()
        self.log_count = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        if self.log_count > 3:
            if "loss" in logs:
                self.train_losses.append(logs["loss"])
                self.train_steps.append(state.global_step)
            if "eval_loss" in logs:
                self.eval_losses.append(logs["eval_loss"])
                self.eval_steps.append(state.global_step)
        
        self.log_count += 1
        self._update_plot()

    def _update_plot(self):
        self.ax.clear()
        if self.train_losses:
            self.ax.plot(self.train_steps, self.train_losses, label="train")
        if self.eval_losses:
            self.ax.plot(self.eval_steps, self.eval_losses, label="eval")
        self.ax.set_xlabel("steps")
        self.ax.set_ylabel("loss")
        self.ax.legend(); self.ax.grid(True)
        self.plot_path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.tight_layout(); self.fig.savefig(self.plot_path)
        plt.pause(0.05)

    def on_train_end(self, *_, **__):
        self.plot_path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(self.plot_path)
        plt.close(self.fig)
        print(f"Loss plot saved to {self.plot_path}")

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
            device_map="auto",
            use_cache=False,               # needed for gradient‑checkpointing
            ignore_mismatched_sizes=True,
        )
        # Add config attribute to make push_to_hub work
        self.config = self.model.config
        # Disable gradient checkpointing for small sequences (40 tokens)


    def forward(self, input_ids, attention_mask=None, labels=None):
        # Forward through base LM and grab column of interest
        logits_full = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        logits_value = logits_full[..., self.value_token_id]  # [B, L]
        newline_mask = torch.zeros_like(input_ids[:, :-1], dtype=torch.bool)
        for token in self.newline_token_ids:
            newline_mask |= (input_ids[:, :-1] == token)
        logits_sel = logits_value[:, :-1][newline_mask]  # [K]
        labels_sel = labels.float().unsqueeze(1).expand_as(newline_mask)[newline_mask]  # [K]
        loss = F.binary_cross_entropy_with_logits(logits_sel, labels_sel, reduction="mean")
        return {"loss": loss, "logits": logits_sel}

# ───────────────────────── Trainer wrapper ────────────────────────── #
class ValueTrainer:
    """High‑level orchestrator: dataset ↔ tokenizer ↔ Trainer."""

    def __init__(self, config: dict):
        # Global performance toggles
        self._apply_speed_optimizations()

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
        print(f"Value token ID for '1': {self.value_token_id}")

        self.model = ValueModel(
            model_name=config["model_name"],
            revision=config["revision"] if config["revision"] != "None" else None,
            newline_token_ids=self.newline_token_ids,
            value_token_id=self.value_token_id
        )


        self.data_collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)
        
        self.temp_dir = tempfile.mkdtemp()

    def _tokenize_split(self, dataset_split):
        def encode(batch):
            return self.tokenizer(batch["text"], truncation=True)

        dataset_split = dataset_split.map(encode, batched=True, remove_columns=["text"])
        dataset_split = dataset_split.rename_column("label", "labels")
        dataset_split.set_format("torch")
        return dataset_split

    def _apply_speed_optimizations(self):
        """Apply speed optimizations for A100 GPUs."""
        # Enable flash kernels
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True  

    def _create_training_args(self):
        return TrainingArguments(
            output_dir=self.temp_dir,

            per_device_train_batch_size=int(self.config["per_device_train_batch_size"]),
            per_device_eval_batch_size=int(self.config["per_device_eval_batch_size"]),
            gradient_accumulation_steps=int(self.config["gradient_accumulation_steps"]),
            num_train_epochs=int(self.config["num_train_epochs"]),
            learning_rate=float(self.config["learning_rate"]),
            lr_scheduler_type=self.config["lr_scheduler_type"],

            logging_steps=int(self.config["logging_steps"]),
            logging_first_step=True,
            eval_strategy="steps",
            eval_steps=int(self.config["eval_steps"]),
            save_steps=int(self.config["eval_steps"]),
            save_total_limit=1,
            metric_for_best_model="eval_loss",
            load_best_model_at_end=True,

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
        dataset = load_from_disk(self.config["dataset_file"])
        train_dataset = self._tokenize_split(dataset["train"].shuffle(seed=42))
        dev_dataset = self._tokenize_split(dataset["dev"].shuffle(seed=42).select(range(min(3000, len(dataset["dev"])))))

        trainer = Trainer(
            model=self.model,
            args=self._create_training_args(),
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=self.data_collator,
            tokenizer=self.tokenizer,
            callbacks=[
                EpochProgressBar(),
                LossScalingCallback(),
                LossPlotCallback(self.config["plot_path"]),
            ],
        )

        trainer.args.train_dataset_size = len(train_dataset)

        print(f"Per device train batch size: {trainer.args.per_device_train_batch_size}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Gradient accumulation steps: {trainer.args.gradient_accumulation_steps}")
        print(f"Effective batch size: {trainer.args.per_device_train_batch_size * max(1, torch.cuda.device_count()) * trainer.args.gradient_accumulation_steps}")
        print(f"Training dataset size: {trainer.args.train_dataset_size}")
        print(f"Steps per epoch: {trainer.args.train_dataset_size // (trainer.args.per_device_train_batch_size * max(1, torch.cuda.device_count()) * trainer.args.gradient_accumulation_steps)}")

        init_metrics = trainer.evaluate()
        print(f"Initial dev loss: {init_metrics['eval_loss']}")

        try:
            trainer.train()
        except KeyboardInterrupt:
            print("Training interrupted – saving current model…")

        print(f"Pushing model to Hugging Face Hub: {self.config['hub_model_id']}")
        try:
            if trainer.state.best_metric < init_metrics['eval_loss']:
                # Try to push to hub first
                try:
                    self.model.model.push_to_hub(self.config["hub_model_id"], push_functional=True)
                    self.tokenizer.push_to_hub(self.config["hub_model_id"])
                    print(f"Model and tokenizer successfully pushed to: https://huggingface.co/{self.config['hub_model_id']}")
                    shutil.rmtree(self.temp_dir)
                except Exception as e:
                    print(f"Error pushing to hub: {e}")
                    print("Saving model locally instead...")
                    # Save the best model parameters locally
                    best_model_path = os.path.join(self.temp_dir, "best_model")
                    os.makedirs(best_model_path, exist_ok=True)
                    self.model.model.save_pretrained(best_model_path)
                    self.tokenizer.save_pretrained(best_model_path)
                    print(f"Best model and tokenizer saved locally to: {best_model_path}")
            else:
                print("Best eval loss is higher than initial dev loss. Not pushing to hub.")
        except Exception as e:
            print(f"Error in model saving process: {e}")


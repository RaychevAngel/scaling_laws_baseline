#!/usr/bin/env python3
"""
-----------------------------------
Assumed dataset format (loaded from disk):

{"text": "foo\nbar\nbaz",        "label": 1}
{"text": "hello\nworld",         "label": 0}
...
--------------------------------------------------
"""

import tempfile
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


class LossScalingCallback(TrainerCallback):
    """Callback to correctly scale the loss when using gradient accumulation."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            # Scale only the training loss by dividing by the number of accumulation steps
            logs["loss"] = logs["loss"] / args.gradient_accumulation_steps

class EpochProgressBar(TrainerCallback):
    def __init__(self):
        self.progress_bar = None
        self.current_epoch = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        if self.progress_bar is not None:
            self.progress_bar.close()
            
        device_count = max(1, torch.cuda.device_count())
        steps_per_epoch = args.train_dataset_size // (args.per_device_train_batch_size * device_count * args.gradient_accumulation_steps)
        
        self.progress_bar = tqdm(
            total=steps_per_epoch,
            desc="Epoch Progress",
            position=0,
            leave=True
        )
        self.current_epoch = 0
    
    def on_step_end(self, args, state, control, **kwargs):
        if self.progress_bar is not None:
            epoch_floor = int(state.epoch)
            if epoch_floor != self.current_epoch:
                self.current_epoch = epoch_floor
                self.progress_bar.reset()
                self.progress_bar.set_description(f"Epoch {epoch_floor + 1}")
            
            fraction_in_epoch = state.epoch - epoch_floor
            current_step = int(fraction_in_epoch * self.progress_bar.total)
            
            self.progress_bar.n = current_step
            self.progress_bar.refresh()
    
    def on_train_end(self, args, state, control, **kwargs):
        if self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None
            
    def on_evaluate(self, args, state, control, **kwargs):
        print(f"\nEvaluating at step {state.global_step} (epoch {state.epoch:.2f})")

class LossPlotCallback(TrainerCallback):
    """Callback to plot training and evaluation loss during training."""
    def __init__(self, plot_path: str):
        self.plot_path = plot_path
        self.train_losses = []
        self.eval_losses = []
        self.train_steps = []
        self.eval_steps = []
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        plt.ion()  # Turn on interactive mode
        self.log_count = 0  # Counter for iterations
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        if self.log_count > 3:
            if "loss" in logs:
                # Collect training loss data
                loss = logs["loss"] 
                self.train_losses.append(loss)
                self.train_steps.append(state.global_step)
                
            if "eval_loss" in logs:
                # Collect evaluation loss data
                self.eval_losses.append(logs["eval_loss"])
                self.eval_steps.append(state.global_step)
            
        self.log_count += 1
        self._update_plot() 
        
        
    def _update_plot(self):
        self.ax.clear()
        if self.train_losses:
            self.ax.plot(self.train_steps, self.train_losses, 'b-', label='Training Loss')
        if self.eval_losses:
            self.ax.plot(self.eval_steps, self.eval_losses, 'r-', label='Evaluation Loss')
            
        self.ax.set_xlabel('Steps')
        self.ax.set_ylabel('Loss')
        self.ax.set_title('Training and Evaluation Loss')
        self.ax.legend()
        self.ax.grid(True)
        
        # Use logarithmic scale if values vary widely
        if self.train_losses and min(self.train_losses) > 0 and max(self.train_losses) / min(self.train_losses) > 10:
            self.ax.set_yscale('log')
            
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)
        
        # Save the plot on each update
        os.makedirs(os.path.dirname(self.plot_path), exist_ok=True)
        plt.savefig(self.plot_path)
        
    def on_train_end(self, args, state, control, **kwargs):
        # Save the final plot
        os.makedirs(os.path.dirname(self.plot_path), exist_ok=True)
        plt.savefig(self.plot_path)
        plt.close(self.fig)
        print(f"Loss plot saved to {self.plot_path}")


class ValueModel(torch.nn.Module):
    """Wrap base LM with custom forward that returns BCE-with-logits loss."""

    def __init__(self, model_name: str, revision: str, newline_id: int, value_token_id: int):
        super().__init__()
        self.newline_id = newline_id
        self.value_token_id = value_token_id  # Token ID corresponding to string "1"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            use_cache=False,  # Required for gradient checkpointing
            ignore_mismatched_sizes=True  # Allow loading with mismatched parameter shapes
        )
        # Enable gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable()

    def forward(self, input_ids, labels=None):
        logits = self.model(input_ids=input_ids).logits[..., self.value_token_id]
        mask_prev_nl = (input_ids == self.newline_id)[:, :-1]
        logits_sel = logits[:, :-1][mask_prev_nl]
        
        if logits_sel.numel() == 0:
            return {"loss": logits.sum() * 0, "logits": logits_sel}
            
        labels_sel = labels.float().unsqueeze(1).expand_as(mask_prev_nl)[mask_prev_nl]
        return {
            "loss": F.binary_cross_entropy_with_logits(logits_sel, labels_sel, reduction="mean"), 
            "logits": logits_sel
        }


class ValueTrainer:
    """Trainer for the Value Model."""
    
    def __init__(self, config):
        # Apply speed optimizations for A100 GPUs
        self._apply_speed_optimizations()
        
        self.config = config
        self.device = config["device"]
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model_name"],
            revision=config["revision"] if config["revision"] != "None" else None  # Handle None string properly
        )
        
        self.newline_id = self.tokenizer.encode("\n", add_special_tokens=False)[0]
        
        # Get the token ID for "1" directly from the tokenizer
        self.value_token_id = self.tokenizer.encode("1", add_special_tokens=False)[0]
        print(f"Value token ID for '1': {self.value_token_id}")
        
        self.model = ValueModel(
            config["model_name"], 
            revision=config["revision"] if config["revision"] != "None" else None,
            newline_id=self.newline_id, 
            value_token_id=self.value_token_id,
        )
        
        # Data collator
        self.collator = DataCollatorWithPadding(self.tokenizer, pad_to_multiple_of=8)
        self.temp_dir = tempfile.mkdtemp()
    
    def _apply_speed_optimizations(self):
        """Apply speed optimizations that would normally be at the global level."""
        # Enable flash kernels
        torch.set_float32_matmul_precision("high")
        
        # Speed optimizations for A100 GPUs
        torch.backends.cuda.matmul.allow_tf32 = True  # Enable TensorFloat32 (A100 specific)
        torch.backends.cudnn.benchmark = True         # Optimize CUDNN for fixed input sizes
        torch.backends.cudnn.deterministic = False    # Allow non-deterministic algorithms if faster
        os.environ["ACCELERATE_USE_FLASH_ATTENTION"] = "true"  # Enable Flash Attention 2
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Consistent device ordering
    
    def _create_training_args(self):
        """Create TrainingArguments from config dictionary."""
        return TrainingArguments(
            output_dir=self.temp_dir,

            per_device_train_batch_size=int(self.config["per_device_train_batch_size"]),
            num_train_epochs=int(self.config["num_train_epochs"]),
            gradient_accumulation_steps=int(self.config["gradient_accumulation_steps"]),
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
            
            # Performance optimizations for 8x A100s
            fp16=False,                      # Use bf16 instead for A100s
            bf16=True,                       # A100s have native bfloat16 support
            dataloader_num_workers=8,        # More workers for A100 throughput
            #group_by_length=True,            # Group similar length sequences for efficiency
            gradient_checkpointing=True,     # Trade compute for memory savings
            ddp_find_unused_parameters=False,# Faster DDP
            ddp_bucket_cap_mb=250,           # Larger bucket size for faster multi-GPU
            tf32=True,                       # Enable TF32 precision (A100-specific)
            
            push_to_hub=True,
            hub_model_id=self.config["hub_model_id"],
            hub_strategy="end",
            hub_private_repo=False,
            disable_tqdm=True,
        )
    
    def train(self):
        dataset = load_from_disk(self.config["dataset_file"])
        
        loss_scaling_callback = LossScalingCallback()
        loss_plot_callback = LossPlotCallback(self.config["plot_path"])

        trainer = Trainer(
            model=self.model,
            train_dataset=dataset["train"].shuffle(seed=42),
            eval_dataset=dataset["dev"].select(range(min(3000, len(dataset["dev"])))).shuffle(seed=42),
            args=self._create_training_args(),
            data_collator=self.collator,
            callbacks=[EpochProgressBar(), loss_scaling_callback, loss_plot_callback]
        )
        
        # Store dataset size for progress bar calculation
        trainer.args.train_dataset_size = len(dataset["train"])
        
        # Print training configuration
        print(f"Per device train batch size: {trainer.args.per_device_train_batch_size}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Gradient accumulation steps: {trainer.args.gradient_accumulation_steps}")
        print(f"Effective batch size: {trainer.args.per_device_train_batch_size * max(1, torch.cuda.device_count()) * trainer.args.gradient_accumulation_steps}")
        print(f"Training dataset size: {trainer.args.train_dataset_size}")
        print(f"Steps per epoch: {trainer.args.train_dataset_size // (trainer.args.per_device_train_batch_size * max(1, torch.cuda.device_count()) * trainer.args.gradient_accumulation_steps)}")
        
        # Train model
        try:
            trainer.train()
        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving and pushing current model to Hugging Face Hub...")

        # Push to hub if configured
        if bool(self.config["push_to_hub"]) and self.config["hub_model_id"]:
            trainer.push_to_hub()
            print(f"Model successfully pushed to: https://huggingface.co/{self.config['hub_model_id']}")

        return trainer

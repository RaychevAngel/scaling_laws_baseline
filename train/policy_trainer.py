import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from datasets import load_from_disk
from trl import SFTConfig, SFTTrainer
import tempfile, shutil, torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ─────────────────────────── Callbacks ──────────────────────────── #
class LossScalingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            logs["loss"] /= args.gradient_accumulation_steps

class LossPlotCallback(TrainerCallback):
    """Callback to plot training loss during training."""
    def __init__(self, plot_path: str):
        self.plot_path = plot_path
        self.train_losses = []
        self.train_steps = []
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        plt.ion()  # Turn on interactive mode
        self.log_count = 0  # Counter for iterations
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        
        if self.log_count > 1:
            if "loss" in logs:
                # Collect training loss data
                loss = logs["loss"] 
                self.train_losses.append(loss)
                self.train_steps.append(state.global_step)
        
        self.log_count += 1
        self._update_plot()
        
    def _update_plot(self):
        self.ax.clear()
        if self.train_losses:
            self.ax.plot(self.train_steps, self.train_losses, 'b-', label='Training Loss')
            
        self.ax.set_xlabel('Steps')
        self.ax.set_ylabel('Loss')
        self.ax.set_title('Training Loss')
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


# ─────────────────────────── Model wrapper ───────────────────────── #
class PolicySFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
        
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        return (loss, None, None) if prediction_loss_only else (loss, outputs.logits, inputs.get("labels"))


# ─────────────────────────── Trainer class ───────────────────────── #
class PolicyTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config["device"]
        
        # Apply speed optimizations
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        
        self.model = AutoModelForCausalLM.from_pretrained(
            config["model_name"], 
            revision=config["revision"] if config["revision"] != "None" else None,
            ignore_mismatched_sizes=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Enable gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model_name"],
            revision=config["revision"] if config["revision"] != "None" else None
        )
        self.temp_dir = tempfile.mkdtemp()
    
    def _create_trainer_config(self):            
        return SFTConfig(
            output_dir=self.temp_dir,

            per_device_train_batch_size=int(self.config["per_device_train_batch_size"]),
            num_train_epochs=1,
            gradient_accumulation_steps=int(self.config["gradient_accumulation_steps"]),
            learning_rate=float(self.config["learning_rate"]),
            lr_scheduler_type=self.config["lr_scheduler_type"],
            
            logging_steps=int(self.config["logging_steps"]),
            logging_first_step=True,
            
            # No intermediate saving
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
        dataset = load_from_disk(self.config["dataset_file"]).shuffle(seed=42)

        callbacks = [
            LossScalingCallback(),
            LossPlotCallback(self.config["plot_path"]),
        ]

        trainer = PolicySFTTrainer(
            model=self.model,
            train_dataset=dataset,
            args=self._create_trainer_config(),
            callbacks=callbacks,
        )
        
        # Provide dataset size to progress bar callback
        trainer.args.train_dataset_size = len(dataset)
        
        # Print training configuration
        print(f"Per device train batch size: {trainer.args.per_device_train_batch_size}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Gradient accumulation steps: {trainer.args.gradient_accumulation_steps}")
        print(f"Effective batch size: {trainer.args.per_device_train_batch_size * max(1, torch.cuda.device_count()) * trainer.args.gradient_accumulation_steps}")
        print(f"Training dataset size: {trainer.args.train_dataset_size}")
        print(f"Steps per epoch: {trainer.args.train_dataset_size // (trainer.args.per_device_train_batch_size * max(1, torch.cuda.device_count()) * trainer.args.gradient_accumulation_steps)}")
        
        # Train the model
        try:
            trainer.train()
            print("Training completed!")
        except KeyboardInterrupt:
            print("\nTraining interrupted.")

        # Push the final model
        print(f"Pushing model to Hugging Face Hub: {self.config['hub_model_id']}")
        try:
            trainer.push_to_hub()
            print(f"Model successfully pushed to: https://huggingface.co/{self.config['hub_model_id']}")
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Error pushing to hub: {e}")
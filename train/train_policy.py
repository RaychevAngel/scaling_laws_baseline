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
        # Apply speed optimizations for A100 GPUs
        self._apply_speed_optimizations()
        
        self.config = config
        self.device = config["device"]
        self.model = AutoModelForCausalLM.from_pretrained(
            config["model_name"], 
            revision=config["revision"] if config["revision"] != "None" else None,
            ignore_mismatched_sizes=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"  # Use device_map instead of manual to(device)
        )
        
        # Enable gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable()
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model_name"],
            revision=config["revision"] if config["revision"] != "None" else None
        )
        self.temp_dir = tempfile.mkdtemp()
    
    def _apply_speed_optimizations(self):
        """Apply speed optimizations for A100 GPUs."""
        # Enable flash kernels
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True         # Optimize CUDNN for fixed input sizes
    
    def _create_trainer_config(self):            
        return SFTConfig(
            output_dir=self.temp_dir,

            per_device_train_batch_size=int(self.config["per_device_train_batch_size"]),
            per_device_eval_batch_size=int(self.config["per_device_eval_batch_size"]),
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
            
            bf16=True,
            dataloader_num_workers=64,
            
            push_to_hub=True,
            hub_model_id=self.config["hub_model_id"],
            hub_strategy="end",
            hub_private_repo=False,
            disable_tqdm=True,
        )
    
    def train(self):
        dataset = load_from_disk(self.config["dataset_file"])
        train_dataset = dataset["train"].rename_columns({"input": "prompt", "output": "completion"})
        dev_dataset = dataset["dev"].rename_columns({"input": "prompt", "output": "completion"})


        loss_scaling_callback = LossScalingCallback()
        loss_plot_callback = LossPlotCallback(self.config["plot_path"])

        trainer = PolicySFTTrainer(
            model=self.model,
            train_dataset=train_dataset.shuffle(seed=42),
            eval_dataset=dev_dataset.select(range(min(3000, len(dev_dataset)))).shuffle(seed=42),
            args=self._create_trainer_config(),
            callbacks=[EpochProgressBar(), loss_scaling_callback, loss_plot_callback]
        )
        
        # Provide dataset size to progress bar callback
        trainer.args.train_dataset_size = len(train_dataset)
        
        # Print training configuration
        print(f"Per device train batch size: {trainer.args.per_device_train_batch_size}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Gradient accumulation steps: {trainer.args.gradient_accumulation_steps}")
        print(f"Effective batch size: {trainer.args.per_device_train_batch_size * max(1, torch.cuda.device_count()) * trainer.args.gradient_accumulation_steps}")
        print(f"Training dataset size: {trainer.args.train_dataset_size}")
        print(f"Steps per epoch: {trainer.args.train_dataset_size // (trainer.args.per_device_train_batch_size * max(1, torch.cuda.device_count()) * trainer.args.gradient_accumulation_steps)}")
        
        # Train and push to hub
        init_metrics = trainer.evaluate()
        print(f"Initial dev loss: {init_metrics['eval_loss']}")
        
        try:
            trainer.train()
            print("Best eval loss:", trainer.state.best_metric)
        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving and pushing current model to Hugging Face Hub...")

        print(f"Pushing model to Hugging Face Hub: {self.config['hub_model_id']}")
        try:
            if trainer.state.best_metric < init_metrics['eval_loss']:
                # Try to push to hub first
                try:
                    trainer.push_to_hub()
                    print(f"Model successfully pushed to: https://huggingface.co/{self.config['hub_model_id']}")
                    shutil.rmtree(self.temp_dir)
                except Exception as e:
                    print(f"Error pushing to hub: {e}")
                    print("Saving model locally instead...")
                    # Save the best model parameters locally
                    best_model_path = os.path.join(self.temp_dir, "best_model")
                    os.makedirs(best_model_path, exist_ok=True)
                    trainer.save_model(best_model_path)
                    print(f"Best model saved locally to: {best_model_path}")
            else:
                print("Best eval loss is higher than initial dev loss. Not pushing to hub.")
        except Exception as e:
            print(f"Error pushing to hub: {e}")
            print("Saving model locally instead...")
            trainer.save_model(os.path.join(self.temp_dir, "policy_final_model"))
            print(f"Model saved locally to: {os.path.join(self.temp_dir, 'policy_final_model')}")
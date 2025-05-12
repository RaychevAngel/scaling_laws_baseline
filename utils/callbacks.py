import os
import torch
from transformers import TrainerCallback
import matplotlib.pyplot as plt
import datetime

class LossPlotCallback(TrainerCallback):
    """Callback to plot training loss during training with hyperparameter display."""
    def __init__(self, plot_path: str, model_type: str):
        self.plot_path = plot_path  # This is the folder path
        self.model_type = model_type  # Model type (Policy, Value, etc.)
        self.train_losses, self.train_steps = [], []
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        plt.ion()  # Turn on interactive mode
        self.log_count = 0  # Counter for iterations
        self.hyperparams = {}
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        
        # Collect hyperparameters on first log
        if self.log_count == 0:
            self.hyperparams = {
                "learning_rate": args.learning_rate,
                "batch_size": args.per_device_train_batch_size,
                "accumulation_steps": args.gradient_accumulation_steps,
                "effective_batch_size": args.per_device_train_batch_size * max(1, torch.cuda.device_count()) * args.gradient_accumulation_steps,
                "dataset_size": getattr(args, "train_dataset_size", "N/A")
            }
        
        self.log_count += 1
        
        # Skip the first 2 logs
        if self.log_count < 10:
            return
            
        if "loss" in logs:
            self.train_losses.append(logs["loss"])
            self.train_steps.append(state.global_step)
        
        self._update_plot()
        
    def _update_plot(self):
        self.ax.clear()
        if self.train_losses:
            self.ax.plot(self.train_steps, self.train_losses, 'b-', label=f'{self.model_type} Training Loss')
            
        self.ax.set_xlabel('Steps')
        self.ax.set_ylabel('Loss')
        self.ax.legend(fontsize=14)
        self.ax.grid(True)
        
        # Use logarithmic scale if values vary widely
        if self.train_losses and min(self.train_losses) > 0 and max(self.train_losses) / min(self.train_losses) > 10:
            self.ax.set_yscale('log')
        
        # Add hyperparameters text with a bigger box
        if self.hyperparams:
            hyperparams_text = '\n'.join([
                f"Learning rate: {self.hyperparams['learning_rate']}",
                f"Batch size: {self.hyperparams['effective_batch_size']}",
                f"Dataset size: {self.hyperparams['dataset_size']}"
            ])
            props = dict(boxstyle='round', facecolor='lightgray', alpha=0.7)
            self.ax.text(0.98, 0.98, hyperparams_text, transform=self.ax.transAxes, 
                     verticalalignment='top', horizontalalignment='right',
                     fontsize=14, bbox=props)
            
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)
        
        # Save the plot on each update with timestamp as filename
        os.makedirs(self.plot_path, exist_ok=True)
        plot_file_path = os.path.join(self.plot_path, f"{self.timestamp}.png")
        plt.savefig(plot_file_path)
        
    def on_train_end(self, args, state, control, **kwargs):
        # Final plot is already saved with the same timestamp
        print(f"Final loss plot saved as: {os.path.join(self.plot_path, f'{self.timestamp}.png')}")
        plt.close(self.fig) 
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, TrainerCallback
from datasets import load_from_disk
from trl import SFTConfig, SFTTrainer
import tempfile, shutil, torch
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

class PolicySFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
        
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        return (loss, None, None) if prediction_loss_only else (loss, outputs.logits, inputs.get("labels"))

class PolicyTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config["device"]
        self.model = AutoModelForCausalLM.from_pretrained(config["model_name"], ignore_mismatched_sizes=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        self.temp_dir = tempfile.mkdtemp()
    
    def _create_trainer_config(self):
        lr_scheduler_kwargs = {}
        if self.config["lr_scheduler_type"] == "reduce_lr_on_plateau":
            lr_scheduler_kwargs = {
                "factor": float(self.config["lr_scheduler_factor"]),
                "patience": int(self.config["lr_scheduler_patience"]),
                "threshold": float(self.config["lr_scheduler_threshold"]),
            }
            
        return SFTConfig(
            output_dir=self.temp_dir,

            auto_find_batch_size=True,
            gradient_accumulation_steps=int(self.config["accumulation_steps"]),
            learning_rate=float(self.config["learning_rate"]),
            lr_scheduler_type=self.config["lr_scheduler_type"],
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            optim=self.config["optimizer"],
            max_grad_norm=float(self.config["max_grad_norm"]),
            weight_decay=float(self.config["weight_decay"]),
            
            logging_strategy="steps",
            logging_steps=int(self.config["logging_steps"]),
            logging_first_step=True,
            eval_strategy="steps",
            eval_steps=int(self.config["eval_steps"]),
            save_strategy="steps",
            save_steps=int(self.config["eval_steps"]),
            save_total_limit=1,
            metric_for_best_model="eval_loss",
            load_best_model_at_end=True,
            
            push_to_hub=True,
            hub_model_id=self.config["hub_model_id"],
            hub_strategy="end",
            hub_private_repo=False,
            disable_tqdm=True,
        )
    
    def train(self):
        dataset = load_from_disk(self.config["dataset_file"])
        
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=int(self.config["patience"]),
            early_stopping_threshold=float(self.config["improvement_tolerance"])
        )

        loss_scaling_callback = LossScalingCallback()

        trainer = PolicySFTTrainer(
            model=self.model,
            train_dataset=dataset["train"].shuffle(seed=42),
            eval_dataset=dataset["dev"].select(range(1000)).shuffle(seed=42),
            args=self._create_trainer_config(),
            callbacks=[EpochProgressBar(), early_stopping_callback, loss_scaling_callback]
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
        
        # Train and push to hub
        trainer.train()
        print(f"Pushing model to Hugging Face Hub: {self.config['hub_model_id']}")
        trainer.push_to_hub()
        print(f"Model successfully pushed to: https://huggingface.co/{self.config['hub_model_id']}")
        shutil.rmtree(self.temp_dir)
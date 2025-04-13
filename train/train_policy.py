from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback, TrainerCallback
from datasets import load_from_disk
from trl import SFTConfig, SFTTrainer
import tempfile
import shutil
from tqdm.auto import tqdm

class EpochProgressBar(TrainerCallback):
    def __init__(self):
        self.step_progress_bar = None
        self.current_step = 0
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        # Initialize or reset the step progress bar for this epoch
        if self.step_progress_bar is not None:
            self.step_progress_bar.close()
        
        # Calculate total steps for this epoch
        dataset_size = len(state.train_dataset)
        batch_size = args.per_device_train_batch_size
        gradient_accumulation_steps = args.gradient_accumulation_steps
        total_steps = dataset_size // (batch_size * gradient_accumulation_steps)
        if dataset_size % (batch_size * gradient_accumulation_steps) != 0:
            total_steps += 1
        
        # Create progress bar for this epoch
        self.step_progress_bar = tqdm(
            total=total_steps,
            desc=f"Epoch {state.epoch:.2f}", 
            position=0
        )
        self.current_step = 0
    
    def on_step_end(self, args, state, control, **kwargs):
        # Update step progress
        if self.step_progress_bar is not None:
            self.current_step += 1
            self.step_progress_bar.update(1)
    
    def on_epoch_end(self, args, state, control, **kwargs):
        # Close progress bar at the end of the epoch
        if self.step_progress_bar is not None:
            self.step_progress_bar.close()
            self.step_progress_bar = None

class PolicyTrainer:
    def __init__(self, config):
        self.config = config
        self.device = self.config["device"]
        # The model will automatically use all available GPUs on the same machine
        self.model = AutoModelForCausalLM.from_pretrained(self.config["model_name"], ignore_mismatched_sizes=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
        self.temp_dir = tempfile.mkdtemp()
    
    def _create_trainer_config(self):
        return SFTConfig(
            output_dir=self.temp_dir,

            gradient_accumulation_steps=int(self.config["accumulation_steps"]),
            learning_rate=float(self.config["learning_rate"]),
            lr_scheduler_type=self.config["lr_scheduler_type"],
            warmup_ratio=float(self.config["warmup_ratio"]),
            optim=self.config["optimizer"],
            max_grad_norm=float(self.config["max_grad_norm"]),

            dropout=float(self.config["dropout"]),
            weight_decay=float(self.config["weight_decay"]),
            
            logging_steps=int(self.config["logging_steps"]),
            eval_steps=int(self.config["eval_steps"]),
            evaluation_strategy="steps",
            save_strategy="no",
            save_best_model=True,
            metric_for_best_model="eval_loss",
            load_best_model_at_end=True,
            
            hub_model_id=self.config["hub_model_id"],
        )
    
    def train(self):
        dataset = load_from_disk(self.config["dataset_file"])
        
        # Create the early stopping callback with min_delta parameter
        early_stopping_callback = EarlyStoppingCallback(
            early_stopping_patience=int(self.config["patience"]),
            min_delta=float(self.config["improvement_tolerance"])
        )
        
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self._create_trainer_config(),
            train_dataset=dataset["train"].shuffle(seed=42),
            eval_dataset=dataset["dev"].shuffle(seed=42),
            callbacks=[EpochProgressBar(), early_stopping_callback],
        )
        
        # Find the optimal batch size for the available GPUs
        trainer.find_executable_batch_size(auto_find_batch_size=True)
        print(f"Batch size: {trainer.args.per_device_train_batch_size}")

        trainer.train()
        print(f"Pushing model to Hugging Face Hub: {self.config['hub_model_id']}")
        trainer.push_to_hub()
        print(f"Model successfully pushed to: https://huggingface.co/{self.config['hub_model_id']}")
        
        shutil.rmtree(self.temp_dir)
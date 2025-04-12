import yaml
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
        self.model = AutoModelForCausalLM.from_pretrained(self.config["model_name"], ignore_mismatched_sizes=True).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
        self.temp_dir = tempfile.mkdtemp()
    
    def _create_trainer_config(self):
        return SFTConfig(
            output_dir=self.temp_dir,

            gradient_accumulation_steps=int(self.config["accumulation_steps"]),
            per_device_train_batch_size=int(self.config["batch_size"]),
            
            learning_rate=float(self.config["learning_rate"]),
            lr_scheduler_type="cosine_with_restarts",
            warmup_ratio=0.1,
            weight_decay=0.01,
            optim="adamw_torch",
            
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
        
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self._create_trainer_config(),
            train_dataset=dataset["train"].shuffle(seed=42),
            eval_dataset=dataset["dev"].shuffle(seed=42),
            callbacks=[EpochProgressBar(), EarlyStoppingCallback(early_stopping_patience=int(self.config["patience"]))],
        )
        
        trainer.train()
        print(f"Pushing model to Hugging Face Hub: {self.config['hub_model_id']}")
        trainer.push_to_hub()
        print(f"Model successfully pushed to: https://huggingface.co/{self.config['hub_model_id']}")
        
        shutil.rmtree(self.temp_dir)
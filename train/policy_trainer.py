import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from trl import SFTConfig, SFTTrainer
import tempfile, shutil, torch
from utils.callbacks import LossPlotCallback

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["model_name"],
            revision=config["revision"] if config["revision"] != "None" else None
        )
        self.temp_dir = tempfile.mkdtemp()
        self.final_loss = None
    
    def _create_trainer_config(self):            
        return SFTConfig(
            output_dir=self.temp_dir,

            per_device_train_batch_size=int(self.config["per_device_train_batch_size"]),
            num_train_epochs=int(self.config["num_train_epochs"]),
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
        callback = LossPlotCallback(self.config["plot_path"], model_type="Policy")

        trainer = SFTTrainer(
            model=self.model,
            train_dataset=load_from_disk(self.config["dataset_file"]).shuffle(seed=42),
            args=self._create_trainer_config(),
            callbacks=[callback],
        )
        
        trainer.args.train_dataset_size = len(trainer.train_dataset)
        effective_batch = trainer.args.per_device_train_batch_size * max(1, torch.cuda.device_count()) * trainer.args.gradient_accumulation_steps
        
        # Print training configuration
        print(f"Per device train batch size: {trainer.args.per_device_train_batch_size}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Gradient accumulation steps: {trainer.args.gradient_accumulation_steps}")
        print(f"Effective batch size: {effective_batch}")
        print(f"Training dataset size: {trainer.args.train_dataset_size}")
        print(f"Steps per epoch: {trainer.args.train_dataset_size // effective_batch}")
        
        # Train the model
        try:
            trainer.train()
            print("Training completed!")
        except KeyboardInterrupt:
            print("\nTraining interrupted.")
            
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
            trainer.push_to_hub()
            print(f"Model successfully pushed to: https://huggingface.co/{self.config['hub_model_id']}")
            shutil.rmtree(self.temp_dir)
        except Exception as e:
            print(f"Error pushing to hub: {e}")
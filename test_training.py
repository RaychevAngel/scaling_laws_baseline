import yaml
from train.train_policy import PolicyTrainer
from train.train_value import ValueTrainer
import os
 # Skip GPU 0, make GPUs 1-7 available if they exist

def main():
    # Load configuration
    with open("train/config_policy.yaml", "r") as f:
        config = yaml.safe_load(f)
    config['model_name'] += str(0)
    config['hub_model_id'] += str(1)
    # Initialize and run training
    trainer = PolicyTrainer(config)
    trainer.train()

    with open("train/config_value.yaml", "r") as f:
        config = yaml.safe_load(f)
    config['model_name'] += str(0)
    config['hub_model_id'] += str(1)
    # Initialize and run training
    trainer = ValueTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()


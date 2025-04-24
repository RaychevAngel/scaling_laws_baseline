import yaml
from train.train_policy import PolicyTrainer
from train.train_value import ValueTrainer
import os

# Skip GPU 0, make GPUs 1-7 available if they exist
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

def main():
    with open("train/config_policy.yaml", "r") as f:
        config1 = yaml.safe_load(f)
    config1['model_name'] += str(0)
    config1['dataset_file'] += str(0)
    config1['plot_path'] += str(0)
    config1['hub_model_id'] += str(1)

    # Initialize and run training
    trainer = PolicyTrainer(config1)
    trainer.train()

    with open("train/config_value.yaml", "r") as f:
        config2 = yaml.safe_load(f)
    config2['model_name'] += str(0)
    config2['dataset_file'] += str(0)
    config2['plot_path'] += str(0)
    config2['hub_model_id'] += str(1)

    # Initialize and run training
    trainer = ValueTrainer(config2)
    trainer.train()

if __name__ == "__main__":
    main()


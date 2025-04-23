import yaml
import os
from train.train_policy import PolicyTrainer
from utils.env_config import get_hf_user
import sys

def main():
    # Check if we're running in profiling mode
    config_path = os.environ.get('PROFILE_CONFIG_PATH')

    # Load configuration from profile path if provided, otherwise use the default
    config_file = config_path if config_path else "train/config_policy.yaml"
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Only replace the hub_model_id for saving, keep model_name as is for loading
    hf_user = get_hf_user()
    if "hub_model_id" in config:
        # Extract model name without username
        parts = config["hub_model_id"].split("/")
        if len(parts) > 1:
            # Construct new hub_model_id with user's username
            config["hub_model_id"] = f"{hf_user}/{parts[1]}"

    # Check if we should limit the number of steps (for profiling)
    max_steps = os.environ.get('MAX_STEPS')
    if max_steps:
        try:
            config['max_steps'] = int(max_steps)
            print(f"Limiting training to {max_steps} steps for profiling")
        except ValueError:
            print(f"Warning: Invalid MAX_STEPS value: {max_steps}")

    # Initialize and run training
    trainer = PolicyTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()


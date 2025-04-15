import yaml
from train.train_policy import PolicyTrainer

def main():
    # Load configuration
    with open("train/config_policy.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize and run training
    trainer = PolicyTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()


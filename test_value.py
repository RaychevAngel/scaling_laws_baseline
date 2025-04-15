import yaml
from train.train_value import ValueTrainer

def main():
    # Load configuration
    with open("train/config_value.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize and run training
    trainer = ValueTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()


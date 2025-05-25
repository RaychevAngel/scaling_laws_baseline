import yaml
import os
from train.policy_trainer import PolicyTrainer

########################################################
i = 6
os.environ["CUDA_VISIBLE_DEVICES"] = "6" 
########################################################s

def main():
    with open("train/config_policy.yaml", "r") as f:
        config_policy = yaml.safe_load(f)

    config_policy['model_name'] += str(i)
    config_policy['dataset_file'] += str(i)
    config_policy['plot_path'] += str(i)
    config_policy['hub_model_id'] += str(i+1)

    config_policy['learning_rate'] = 5e-6 
    config_policy['per_device_train_batch_size'] = 128
    config_policy['gradient_accumulation_steps'] = 4
    config_policy['logging_steps'] = 1

    policy_trainer = PolicyTrainer(config_policy)
    policy_trainer.train()

if __name__ == "__main__":
    main()








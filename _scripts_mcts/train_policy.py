import yaml
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--iter", type=int, required=True)
parser.add_argument("--gpu", type=int, required=True)
args = parser.parse_args()

########################################################
checkpoint = args.iter
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) 
########################################################s

from train.policy_trainer import PolicyTrainer

def main():
    with open("train/config_policy.yaml", "r") as f:
        config_policy = yaml.safe_load(f)

    config_policy['model_name'] += str(checkpoint)
    config_policy['dataset_file'] += str(checkpoint)
    config_policy['plot_path'] += str(checkpoint)
    config_policy['hub_model_id'] += str(checkpoint+1)

    config_policy['learning_rate'] = 5e-5
    config_policy['per_device_train_batch_size'] = 64
    config_policy['gradient_accumulation_steps'] = 4
    config_policy['num_train_epochs'] = 1

    policy_trainer = PolicyTrainer(config_policy)
    policy_trainer.train()

if __name__ == "__main__":
    main()








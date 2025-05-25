import yaml
import os
from train.policy_trainer import PolicyTrainer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--iter", type=int, required=True)
parser.add_argument("--gpu", type=int, required=True)
parser.add_argument("--epochs", type=int, required=True)
args = parser.parse_args()

########################################################
checkpoint = args.iter
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) 
########################################################s

def main():
    with open("train/config_sos.yaml", "r") as f:
        config_sos = yaml.safe_load(f)

    config_sos['model_name'] += str(checkpoint)
    config_sos['dataset_file'] += str(checkpoint)
    config_sos['plot_path'] += str(checkpoint)
    config_sos['hub_model_id'] += str(checkpoint+1)

    config_sos['learning_rate'] = 5e-6 
    config_sos['per_device_train_batch_size'] = 16
    config_sos['gradient_accumulation_steps'] = 4
    config_sos['logging_steps'] = 1
    config_sos['num_train_epochs'] = args.epochs

    sos_trainer = PolicyTrainer(config_sos)
    sos_trainer.train()

if __name__ == "__main__":
    main()








import os
import argparse
import yaml

# Parse args first to get GPU setting
parser = argparse.ArgumentParser()
parser.add_argument("--iter", type=int, required=True)
parser.add_argument("--gpu", type=int, required=True)
args = parser.parse_args()

########################################################
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) 
checkpoint = args.iter
########################################################s

from train.value_trainer import ValueTrainer

def main():
    with open("train/config_value.yaml", "r") as f:
        config_value = yaml.safe_load(f)

    config_value['model_name'] += str(checkpoint)
    config_value['dataset_file'] += str(checkpoint)
    config_value['plot_path'] += str(checkpoint)
    config_value['hub_model_id'] += str(checkpoint+1)

    config_value['learning_rate'] = 2e-5
    config_value['per_device_train_batch_size'] = 128
    config_value['gradient_accumulation_steps'] = 8
    config_value['num_train_epochs'] = 1

    value_trainer = ValueTrainer(config_value)
    value_trainer.train()

if __name__ == "__main__":
    main()








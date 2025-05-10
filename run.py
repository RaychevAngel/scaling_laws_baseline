import yaml
import asyncio
import os
from train.train_policy import PolicyTrainer
from train.train_value import ValueTrainer
from generate_data.generate_data import DataGenerator
import torch
import gc

async def data_generation(config, policy_gpu, value_gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    data_generator = DataGenerator(config, policy_gpu, value_gpu)
    await data_generator.run()
    del data_generator
    gc.collect()
    torch.cuda.empty_cache()

async def train_policy(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    policy_trainer = PolicyTrainer(config)
    policy_trainer.train()
    del policy_trainer
    gc.collect()
    torch.cuda.empty_cache()

async def train_value(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    value_trainer = ValueTrainer(config)
    value_trainer.train()
    del value_trainer
    gc.collect()
    torch.cuda.empty_cache()


async def main(num_iterations, policy_gpu_gen, value_gpu_gen):
    # Load configuration
    with open("train/config_policy.yaml", "r") as f:
        config_policy = yaml.safe_load(f)
    with open("train/config_value.yaml", "r") as f:
        config_value = yaml.safe_load(f)
    with open("generate_data/config_mcts_generator.yaml", "r") as f:
        config_mcts_generator = yaml.safe_load(f)

    # Store base paths/names
    base_paths = {
        'gen_policy_model': config_mcts_generator['policy_model'],
        'gen_value_model': config_mcts_generator['value_model'],
        'policy_data_path': config_mcts_generator['policy_data_path'],
        'value_data_path': config_mcts_generator['value_data_path'],
        'train_questions_path': config_mcts_generator['train_questions_path'],

        'policy_model_name': config_policy['model_name'],
        'policy_dataset_file': config_policy['dataset_file'],
        'policy_hub_model_id': config_policy['hub_model_id'],
        'policy_plot_path': config_policy['plot_path'],

        'value_model_name': config_value['model_name'],
        'value_dataset_file': config_value['dataset_file'],
        'value_hub_model_id': config_value['hub_model_id'],
        'value_plot_path': config_value['plot_path']
    }

    for i in range(1, num_iterations):
        # Update generator config with iteration numbers
        config_mcts_generator['policy_model'] = base_paths['gen_policy_model'] + str(i)
        config_mcts_generator['value_model'] = base_paths['gen_value_model'] + str(i)
        config_mcts_generator['policy_data_path'] = base_paths['policy_data_path'] + str(i)
        config_mcts_generator['value_data_path'] = base_paths['value_data_path'] + str(i)
        config_mcts_generator['train_questions_path'] = base_paths['train_questions_path'] + str(i) + ".txt"

        # Run MCTS data generation with proper GPUs
        #await data_generation(config_mcts_generator, policy_gpu_gen, value_gpu_gen)

        # Update policy trainer config
        config_policy['model_name'] = base_paths['policy_model_name'] + str(i)
        config_policy['dataset_file'] = base_paths['policy_dataset_file'] + str(i)
        config_policy['plot_path'] = base_paths['policy_plot_path'] + str(i)
        config_policy['hub_model_id'] = base_paths['policy_hub_model_id'] + str(i+1)
        
        # Set GPU for policy training
        await train_policy(config_policy)
        
        # Update value trainer config
        config_value['model_name'] = base_paths['value_model_name'] + str(i)
        config_value['dataset_file'] = base_paths['value_dataset_file'] + str(i)
        config_value['plot_path'] = base_paths['value_plot_path'] + str(i)
        config_value['hub_model_id'] = base_paths['value_hub_model_id'] + str(i+1)
        
        # Using same GPU for value training (redundant but explicit for clarity)
        await train_value(config_value)


if __name__ == "__main__":
    asyncio.run(main(num_iterations=10, policy_gpu_gen=0, value_gpu_gen=1))


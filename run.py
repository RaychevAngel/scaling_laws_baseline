import yaml
import asyncio
import os
from train.train_policy import PolicyTrainer
from train.train_value import ValueTrainer
from generate_data.generate_data import DataGenerator
from evaluate.evaluate import Evaluator


async def main(num_iterations, policy_gpu_eval, value_gpu_eval,
        policy_gpu_gen, value_gpu_gen):
    # Load configuration
    with open("train/config_policy.yaml", "r") as f:
        config_policy = yaml.safe_load(f)
    with open("train/config_value.yaml", "r") as f:
        config_value = yaml.safe_load(f)
    with open("generate_data/config_mcts_generator.yaml", "r") as f:
        config_mcts_generator = yaml.safe_load(f)
    with open("evaluate/config_mcts_evaluator.yaml", "r") as f:
        config_mcts_evaluator = yaml.safe_load(f)

    # Store base paths/names
    base_paths = {
        'policy_model': config_mcts_evaluator['policy_model'],
        'value_model': config_mcts_evaluator['value_model'],
        'export_data_path': config_mcts_evaluator['export_data_path'],
        'gen_policy_model': config_mcts_generator['policy_model'],
        'gen_value_model': config_mcts_generator['value_model'],
        'policy_data_path': config_mcts_generator['policy_data_path'],
        'value_data_path': config_mcts_generator['value_data_path'],
        'policy_model_name': config_policy['model_name'],
        'policy_dataset_file': config_policy['dataset_file'],
        'policy_hub_model_id': config_policy['hub_model_id'],
        'value_model_name': config_value['model_name'],
        'value_dataset_file': config_value['dataset_file'],
        'value_hub_model_id': config_value['hub_model_id'],
        'policy_plot_path': config_policy['plot_path'],
        'value_plot_path': config_value['plot_path']
    }

    for i in range(2, num_iterations + 1):
        # Update evaluator config with iteration numbers
        config_mcts_evaluator['policy_model'] = base_paths['policy_model'] + str(i)
        config_mcts_evaluator['value_model'] = base_paths['value_model'] + str(i)
        config_mcts_evaluator['export_data_path'] = base_paths['export_data_path'] + str(i)
        config_mcts_evaluator['test_questions_path'] = "questions/dev.txt"
        
        evaluator = Evaluator(config_mcts_evaluator, policy_gpu_eval, value_gpu_eval)
        await evaluator.run()

        # Update generator config with iteration numbers
        config_mcts_generator['policy_model'] = base_paths['gen_policy_model'] + str(i)
        config_mcts_generator['value_model'] = base_paths['gen_value_model'] + str(i)
        config_mcts_generator['policy_data_path'] = base_paths['policy_data_path'] + str(i)
        config_mcts_generator['value_data_path'] = base_paths['value_data_path'] + str(i)
        
        # Run MCTS data generation
        data_generator = DataGenerator(config_mcts_generator, policy_gpu_gen, value_gpu_gen)
        await data_generator.run()
        
        # Update policy and value trainer configs
        config_policy['model_name'] = base_paths['policy_model_name'] + str(i)
        config_policy['dataset_file'] = base_paths['policy_dataset_file'] + str(i)
        config_policy['plot_path'] = base_paths['policy_plot_path'] + str(i)
        config_policy['hub_model_id'] = base_paths['policy_hub_model_id'] + str(i+1)
        
        # Initialize and run training
        policy_trainer = PolicyTrainer(config_policy)
        policy_trainer.train()
        
        config_value['model_name'] = base_paths['value_model_name'] + str(i)
        config_value['dataset_file'] = base_paths['value_dataset_file'] + str(i)
        config_value['plot_path'] = base_paths['value_plot_path'] + str(i)
        config_value['hub_model_id'] = base_paths['value_hub_model_id'] + str(i+1)
        
        # Initialize and run training
        value_trainer = ValueTrainer(config_value)
        value_trainer.train()

if __name__ == "__main__":
    asyncio.run(main(1, 0, 1, 0, 1))


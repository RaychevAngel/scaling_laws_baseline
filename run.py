import yaml
import asyncio
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


    for i in range(1, num_iterations + 1):
        config_mcts_evaluator['policy_model'] += str(i)
        config_mcts_evaluator['value_model'] += str(i)
        config_mcts_evaluator['export_data_path'] += str(i)
        
        evaluator = Evaluator(config_mcts_evaluator, policy_gpu_eval, value_gpu_eval)
        await evaluator.run()

        config_mcts_generator['policy_model'] += str(i)
        config_mcts_generator['value_model'] += str(i)
        config_mcts_generator['policy_data_path'] += str(i)
        config_mcts_generator['value_data_path'] += str(i)
        # Run MCTS data generation
        data_generator = DataGenerator(config_mcts_generator, policy_gpu_gen, value_gpu_gen)
        await data_generator.run()
        
        config_policy['model_name'] += str(i)
        config_policy['dataset_file'] += str(i)
        config_policy['hub_model_id'] += str(i+1)
        config_value['model_name'] += str(i)
        config_value['dataset_file'] += str(i)
        config_value['hub_model_id'] += str(i+1)
        # Initialize and run training
        policy_trainer = PolicyTrainer(config_policy)
        policy_trainer.train()
        # Initialize and run training
        value_trainer = ValueTrainer(config_value)
        value_trainer.train()

if __name__ == "__main__":
    asyncio.run(main(2, 0, 1, 0, 1))


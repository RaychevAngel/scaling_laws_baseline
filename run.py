import yaml
import asyncio
from train.train_policy import PolicyTrainer
from train.train_value import ValueTrainer
from generate_data.generate_data import DataGenerator

async def main(num_iterations, policy_gpu, value_gpu):
    # Load configuration
    with open("train/config_policy.yaml", "r") as f:
        config_policy = yaml.safe_load(f)
    with open("train/config_value.yaml", "r") as f:
        config_value = yaml.safe_load(f)
    with open("generate_data/config_mcts_generator.yaml", "r") as f:
        config_mcts_generator = yaml.safe_load(f)


    for i in range(1, num_iterations + 1):
        config_mcts_generator['policy_model'] += str(i)
        config_mcts_generator['value_model'] += str(i)
        config_mcts_generator['policy_data_path'] += str(i)
        config_mcts_generator['value_data_path'] += str(i)
        # Run MCTS data generation
        data_generator = DataGenerator(config_mcts_generator, policy_gpu, value_gpu)
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
    asyncio.run(main(2, 6, 7))


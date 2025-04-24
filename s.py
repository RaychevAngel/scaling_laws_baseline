import requests
from utils.policy_value import PolicyValueFunction
from generate_data.mcts_generator import RunMCTS_Generate
from evaluate.mcts_evaluator import RunMCTS_Evaluate
import yaml
import asyncio
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

async def main():
    with open('evaluate/config_mcts_evaluator.yaml', 'r') as f:
        evaluate_config = yaml.safe_load(f)

    evaluate_config['policy_port'] = 8050
    evaluate_config['value_port'] = 8051

    for key in ['policy_model', 'value_model', 'export_data_path']:
        evaluate_config[key] += str(2)

    evaluate_config['test_questions_path'] = "questions/dev.txt"

    policy_value_fn = PolicyValueFunction(evaluate_config)

    

    await RunMCTS_Evaluate(evaluate_config, policy_value_fn).run()
    
if __name__ == "__main__":
    asyncio.run(main())
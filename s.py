import requests
from utils.policy_value import PolicyValueFunction
from generate_data.mcts_generator import RunMCTS_Generate
from evaluate.mcts_evaluator import RunMCTS_Evaluate
import yaml
import asyncio
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

async def main():
    with open('evaluate/config_mcts_evaluator.yaml', 'r') as f:
        evaluate_config = yaml.safe_load(f)

    evaluate_config['policy_port'] = 8056
    evaluate_config['value_port'] = 8057

    for key in ['policy_model', 'value_model', 'export_data_path']:
        evaluate_config[key] += str(4)

    evaluate_config['test_questions_path'] = "questions/dev.txt"

    for branch_factor, max_expansions in [(2, 16), (3, 11)]:
        evaluate_config['branch_factor'] = branch_factor
        evaluate_config['max_expansions'] = max_expansions
        policy_value_fn = PolicyValueFunction(evaluate_config)
        await RunMCTS_Evaluate(evaluate_config, policy_value_fn).run()
    
    for branch_factor, max_expansions in [(3, 21), (4, 16), (5, 13), (6, 11), (7, 9)]:
        evaluate_config['branch_factor'] = branch_factor
        evaluate_config['max_expansions'] = max_expansions
        policy_value_fn = PolicyValueFunction(evaluate_config)
        await RunMCTS_Evaluate(evaluate_config, policy_value_fn).run()

    for branch_factor, max_expansions in [(4, 32), (5, 26), (6, 21), (7, 18), (8, 16), (9, 14)]:
        evaluate_config['branch_factor'] = branch_factor
        evaluate_config['max_expansions'] = max_expansions
        policy_value_fn = PolicyValueFunction(evaluate_config)
        await RunMCTS_Evaluate(evaluate_config, policy_value_fn).run()
    
    for branch_factor, max_expansions in [(5, 51), (6, 43), (7, 37), (8, 32), (9, 28), (10, 26)]:
        evaluate_config['branch_factor'] = branch_factor
        evaluate_config['max_expansions'] = max_expansions
        policy_value_fn = PolicyValueFunction(evaluate_config)
        await RunMCTS_Evaluate(evaluate_config, policy_value_fn).run()

    for branch_factor, max_expansions in [(6, 85), (8, 64), (10, 51), (12, 43), (14, 37), (16, 32)]:
        evaluate_config['branch_factor'] = branch_factor
        evaluate_config['max_expansions'] = max_expansions
        policy_value_fn = PolicyValueFunction(evaluate_config)
        await RunMCTS_Evaluate(evaluate_config, policy_value_fn).run()

    for branch_factor, max_expansions in [(10, 102), (13, 79), (16, 64), (19, 54)]:
        evaluate_config['branch_factor'] = branch_factor
        evaluate_config['max_expansions'] = max_expansions
        policy_value_fn = PolicyValueFunction(evaluate_config)
        await RunMCTS_Evaluate(evaluate_config, policy_value_fn).run()
    
if __name__ == "__main__":
    asyncio.run(main())
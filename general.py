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
        evaluate_config[key] += str(3)
    #evaluate_config['policy_model'] = "AngelRaychev/0.5B-policy-iteration_3"

    evaluate_config['test_questions_path'] = "questions/dev.txt"
    evaluate_config['batch_size'] = 20
    evaluate_config['stats_interval'] = 30

    #for branch_factor, max_expansions in []: #(1, 4), (2, 8)
    #    evaluate_config['branch_factor'] = branch_factor
    #    evaluate_config['max_expansions'] = max_expansions
    #    policy_value_fn = PolicyValueFunction(evaluate_config)
    #    await RunMCTS_Evaluate(evaluate_config, policy_value_fn).run()

    #for branch_factor, max_expansions in []: #(3, 11), (4, 8)
    #    evaluate_config['branch_factor'] = branch_factor
    #    evaluate_config['max_expansions'] = max_expansions
    #    policy_value_fn = PolicyValueFunction(evaluate_config)
    #    await RunMCTS_Evaluate(evaluate_config, policy_value_fn).run()
    
    #for branch_factor, max_expansions in []: #(3, 21), (4, 16), (5, 13), (6, 11), (7, 9)
    #    evaluate_config['branch_factor'] = branch_factor
    #    evaluate_config['max_expansions'] = max_expansions
    #    policy_value_fn = PolicyValueFunction(evaluate_config)
    #    await RunMCTS_Evaluate(evaluate_config, policy_value_fn).run()

    #for branch_factor, max_expansions in []: #(4, 32), (5, 26), (6, 21), (7, 18), (8, 16), (9, 14), (10,13)
    #    evaluate_config['branch_factor'] = branch_factor
    #    evaluate_config['max_expansions'] = max_expansions
    #    policy_value_fn = PolicyValueFunction(evaluate_config)
    #    await RunMCTS_Evaluate(evaluate_config, policy_value_fn).run()
    
    #for branch_factor, max_expansions in [(14, 18)]: #(8, 32), (9, 28), (10, 26), (11, 23), (12, 21), (13, 20), (14, 18)
    #    evaluate_config['branch_factor'] = branch_factor
    #    evaluate_config['max_expansions'] = max_expansions
    #    policy_value_fn = PolicyValueFunction(evaluate_config)
    #    await RunMCTS_Evaluate(evaluate_config, policy_value_fn).run()

    for branch_factor, max_expansions in [(20, 26)]: #(6, 85), (8, 64), (10, 51), (12, 43), (14, 37), (16, 32), (18, 28), (20, 26)
        evaluate_config['branch_factor'] = branch_factor
        evaluate_config['max_expansions'] = max_expansions
        policy_value_fn = PolicyValueFunction(evaluate_config)
        await RunMCTS_Evaluate(evaluate_config, policy_value_fn).run()

    for branch_factor, max_expansions in [(28, 37)]: #(10, 102), (13, 79), (16, 64), (19, 54) (22, 47) (25, 41) (28, 37)
        evaluate_config['branch_factor'] = branch_factor
        evaluate_config['max_expansions'] = max_expansions
        policy_value_fn = PolicyValueFunction(evaluate_config)
        await RunMCTS_Evaluate(evaluate_config, policy_value_fn).run()
    
if __name__ == "__main__":
    asyncio.run(main())
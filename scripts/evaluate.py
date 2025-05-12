from utils.policy_value import PolicyValueFunction
from evaluate.mcts_evaluator import RunMCTS_Evaluate
import yaml
import asyncio

########################################################
i = 3
policy_port = 8052
value_port = 8053
########################################################

async def main():
    with open('evaluate/config_mcts_evaluator.yaml', 'r') as f:
        evaluate_config = yaml.safe_load(f)

    evaluate_config['policy_model'] += str(i)
    evaluate_config['value_model'] += str(i)
    evaluate_config['export_data_path'] += str(i)

    evaluate_config['policy_port'] = policy_port
    evaluate_config['value_port'] = value_port
    
    
    evaluate_config['temperature'] = 1.0
    evaluate_config['c_explore'] = 0.3

    forward_passes = 200
    
    ########################################################
    compute_16 = [(2, 8), (3, 5)]
    for branch_factor, max_expansions in []:
        evaluate_config['batch_size'] = int(forward_passes / branch_factor)
        evaluate_config['branch_factor'] = branch_factor
        evaluate_config['max_expansions'] = max_expansions
        policy_value_fn = PolicyValueFunction(evaluate_config)
        await RunMCTS_Evaluate(evaluate_config, policy_value_fn).run()

    compute_32 = [(3, 11), (4, 8)]
    for branch_factor, max_expansions in []:
        evaluate_config['batch_size'] = int(forward_passes / branch_factor)
        evaluate_config['branch_factor'] = branch_factor
        evaluate_config['max_expansions'] = max_expansions
        policy_value_fn = PolicyValueFunction(evaluate_config)
        await RunMCTS_Evaluate(evaluate_config, policy_value_fn).run()
    
    compute_64 = [(2,32), (3, 21), (4, 16), (5, 13), (6, 11), (7, 9)]
    for branch_factor, max_expansions in []:
        evaluate_config['batch_size'] = int(forward_passes / branch_factor)
        evaluate_config['branch_factor'] = branch_factor
        evaluate_config['max_expansions'] = max_expansions
        policy_value_fn = PolicyValueFunction(evaluate_config)
        await RunMCTS_Evaluate(evaluate_config, policy_value_fn).run()

    compute_128 = [(4, 32), (5, 26), (6, 21), (7, 18), (8, 16), (9, 14), (10,13)]
    for branch_factor, max_expansions in [(5, 26)]:
        evaluate_config['batch_size'] = int(forward_passes / branch_factor)
        evaluate_config['branch_factor'] = branch_factor
        evaluate_config['max_expansions'] = max_expansions
        policy_value_fn = PolicyValueFunction(evaluate_config)
        await RunMCTS_Evaluate(evaluate_config, policy_value_fn).run()
    
    compute_256 = [(8, 32), (9, 28), (10, 26), (11, 23), (12, 21), (13, 20), (14, 18)]
    for branch_factor, max_expansions in []:
        evaluate_config['batch_size'] = int(forward_passes / branch_factor)
        evaluate_config['branch_factor'] = branch_factor
        evaluate_config['max_expansions'] = max_expansions
        policy_value_fn = PolicyValueFunction(evaluate_config)
        await RunMCTS_Evaluate(evaluate_config, policy_value_fn).run()

    compute_512 = [(6, 85), (8, 64), (10, 51), (12, 43), (14, 37), (16, 32), (18, 28), (20, 26)]
    for branch_factor, max_expansions in []:
        evaluate_config['batch_size'] = int(forward_passes / branch_factor)
        evaluate_config['branch_factor'] = branch_factor
        evaluate_config['max_expansions'] = max_expansions
        policy_value_fn = PolicyValueFunction(evaluate_config)
        await RunMCTS_Evaluate(evaluate_config, policy_value_fn).run()

    compute_1024 = [(10, 102), (13, 79), (16, 64), (19, 54), (22, 47), (25, 41), (28, 37)]
    for branch_factor, max_expansions in []:
        evaluate_config['batch_size'] = int(forward_passes / branch_factor)
        evaluate_config['branch_factor'] = branch_factor
        evaluate_config['max_expansions'] = max_expansions
        policy_value_fn = PolicyValueFunction(evaluate_config)
        await RunMCTS_Evaluate(evaluate_config, policy_value_fn).run()
    
if __name__ == "__main__":
    asyncio.run(main())
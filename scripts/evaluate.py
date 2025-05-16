from utils.policy_value import PolicyValueFunction
from evaluate.mcts_evaluator import RunMCTS_Evaluate
import yaml
import asyncio

########################################################
i = 7
k = 13
########################################################
iteration = i 
policy_port = 8050 +2*k
value_port = 8050 + 2*k + 1
########################################################
def print_config(evaluate_config):
    for key in evaluate_config.keys():
        print(key, evaluate_config[key])


async def main():
    with open('evaluate/config_mcts_evaluator.yaml', 'r') as f:
        evaluate_config = yaml.safe_load(f)

    evaluate_config['policy_model'] += str(iteration)
    evaluate_config['value_model'] += str(iteration)
    evaluate_config['export_data_path'] += str(iteration)

    evaluate_config['policy_port'] = policy_port
    evaluate_config['value_port'] = value_port
    
    evaluate_config['test_questions_path'] = "questions/dev.txt"
    
    evaluate_config['temperature'] = 1.0
    evaluate_config['c_explore'] = 0.3

    forward_passes = 300

    ########################################################
    list_11 = [(11, 33), (11, 39), (11, 44), (11, 55)]
    list_12 = [(12, 36), (12, 42), (12, 48), (12, 60)]
    list_13 = [(13, 39), (13, 46), (13, 52), (13, 65)]
    for branch_factor, max_expansions in [(35, 85)]:
        evaluate_config['batch_size'] = int(forward_passes / branch_factor)
        evaluate_config['branch_factor'] = branch_factor
        evaluate_config['max_expansions'] = max_expansions
        policy_value_fn = PolicyValueFunction(evaluate_config)
        print_config(evaluate_config)
        await RunMCTS_Evaluate(evaluate_config, policy_value_fn).run()
    
if __name__ == "__main__":
    asyncio.run(main())
from utils.policy_value import PolicyValueFunction
from evaluate.mcts_evaluator import RunMCTS_Evaluate
import yaml
import asyncio
import argparse
import ast

parser = argparse.ArgumentParser()
parser.add_argument("--iter", type=int, required=True)
parser.add_argument("--gpu", type=int, required=True)
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--be", type=str, required=True, help="Branch factor and max expansions as a string")
parser.add_argument("--mode",type=str, required=True, choices=["eval_train", "eval_dev", "eval_test"])
args = parser.parse_args()

########################################################
checkpoint = args.iter 
policy_port = 8050 + 4*args.gpu + 2*args.port
value_port = 8050 + 4*args.gpu + 2*args.port + 1
mcts_questions_path = (f"questions/train_{4*(args.iter-1)}.txt" if args.mode == "eval_train" 
                     else f"questions/dev.txt" if args.mode == "eval_dev" 
                     else f"questions/test.txt" if args.mode == "eval_test" 
                     else None)
########################################################

def print_config(evaluate_config):
    for key in evaluate_config.keys():
        print(key, evaluate_config[key])


async def main():
    with open('evaluate/config_mcts_evaluator.yaml', 'r') as f:
        evaluate_config = yaml.safe_load(f)

    evaluate_config['policy_model'] += str(checkpoint)
    evaluate_config['value_model'] += str(checkpoint)
    evaluate_config['export_data_path'] += str(checkpoint)
    evaluate_config['policy_port'] = policy_port
    evaluate_config['value_port'] = value_port
    evaluate_config['questions_path'] = mcts_questions_path

    evaluate_config['stats_interval'] = 10
    evaluate_config['c_explore'] = 0.3

    forward_passes = 300
    
    # Convert string representation to list of tuples
    branch_expansions = ast.literal_eval(args.be)

    for branch_factor, max_expansions in branch_expansions:
        evaluate_config['batch_size'] = 300 #int(forward_passes / branch_factor)
        evaluate_config['branch_factor'] = branch_factor
        evaluate_config['max_expansions'] = max_expansions
        policy_value_fn = PolicyValueFunction(evaluate_config)
        print_config(evaluate_config)
        await RunMCTS_Evaluate(evaluate_config, policy_value_fn).run()
    
if __name__ == "__main__":
    asyncio.run(main())
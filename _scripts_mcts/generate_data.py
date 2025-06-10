from utils.policy_value import PolicyValueFunction
from generate_data.mcts_generator import RunMCTS_Generate
import yaml
import asyncio
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--iter", type=int, required=True)
parser.add_argument("--gpu", type=int, required=True)
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--b", type=str, required=True)
parser.add_argument("--e", type=int, required=True)
args = parser.parse_args()


########################################################    
checkpoint = args.iter 
policy_port = 8050 + 4*args.gpu + 2*args.port
value_port = 8050 + 4*args.gpu + 2*args.port + 1
branch_factor = int(args.b)
max_expansions = int(args.e)
########################################################

async def main():
    with open('generate_data/config_mcts_generator.yaml', 'r') as f:
        generate_config = yaml.safe_load(f)

    generate_config['policy_model'] += str(checkpoint)
    generate_config['value_model'] += str(checkpoint)
    generate_config['policy_data_path'] += str(checkpoint)
    generate_config['value_data_path'] += str(checkpoint)
    generate_config['sos_data_path'] += f"_b{branch_factor}_e{max_expansions}"
    generate_config['questions_path'] += str(4*checkpoint + args.gpu) + ".txt"

    generate_config['policy_port'] = policy_port
    generate_config['value_port'] = value_port
    
    generate_config['branch_factor'] = branch_factor
    generate_config['max_expansions'] = [max_expansions]
    generate_config['stats_interval'] = 60

    generate_config['batch_size'] = int(200 / branch_factor)

    for key in generate_config.keys():
        print(key, generate_config[key])

    policy_value_fn = PolicyValueFunction(generate_config)
    await RunMCTS_Generate(generate_config, policy_value_fn).run()

    
if __name__ == "__main__":
    asyncio.run(main())
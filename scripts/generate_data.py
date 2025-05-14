from utils.policy_value import PolicyValueFunction
from generate_data.mcts_generator import RunMCTS_Generate
import yaml
import asyncio

########################################################    
i = 3
k = 7
policy_port = 8050 + 2*k
value_port = 8050 + 2*k + 1
########################################################

async def main():
    with open('generate_data/config_mcts_generator.yaml', 'r') as f:
        generate_config = yaml.safe_load(f)

    generate_config['policy_model'] += str(i)
    generate_config['value_model'] += str(i)
    generate_config['policy_data_path'] += str(i)
    generate_config['value_data_path'] += str(i)
    generate_config['train_questions_path'] += str(4*i + k%4) + ".txt"

    generate_config['policy_port'] = policy_port
    generate_config['value_port'] = value_port
    
    generate_config['branch_factor'] = 5
    generate_config['max_expansions'] = 26
    generate_config['temperature'] = 1.0
    generate_config['c_explore'] = 0.3

    forward_passes = 200
    generate_config['batch_size'] = int(forward_passes / generate_config['branch_factor'])

    for key in generate_config.keys():
        print(key, generate_config[key])

    policy_value_fn = PolicyValueFunction(generate_config)
    await RunMCTS_Generate(generate_config, policy_value_fn).run()

    
if __name__ == "__main__":
    asyncio.run(main())
import requests
from utils.policy_value import PolicyValueFunction
from generate_data.mcts_generator import RunMCTS_Generate
import yaml
import asyncio
async def main():
    with open('generate_data/config_mcts_generator.yaml', 'r') as f:
        config = yaml.safe_load(f)
    config['policy_port'] = 8031
    config['value_port'] = 8027

    for key in ['policy_data_path', 'value_data_path', 'policy_model', 'value_model']:
        config[key] += str(1)

    for i in [8026, 8028, 8031, 8030, 8032]:
        config['policy_port'] = i
        print(config['policy_port'])
        policy_value_fn = PolicyValueFunction(config)

        qs = [('Use 2, 4, 6, 6 to make 26.\n', "")]
        print(qs[0][0])
        result = policy_value_fn(qs)
        for r in result:
            for k in r:
                print(k[0])
    #await RunMCTS_Generate(config, policy_value_fn).run()

if __name__ == "__main__":
    asyncio.run(main())
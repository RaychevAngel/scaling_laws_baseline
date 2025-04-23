import requests
from utils.policy_value import PolicyValueFunction
from generate_data.mcts_generator import RunMCTS_Generate
from evaluate.mcts_evaluator import RunMCTS_Evaluate
import yaml
import asyncio

async def main():
    with open('generate_data/config_mcts_generator.yaml', 'r') as f:
        generate_config = yaml.safe_load(f)
    with open('evaluate/config_mcts_evaluator.yaml', 'r') as f:
        evaluate_config = yaml.safe_load(f)
    generate_config['policy_port'] = 8032
    generate_config['value_port'] = 8027

    for key in ['policy_data_path', 'value_data_path', 'policy_model', 'value_model']:
        generate_config[key] += str(1)
    for key in ['policy_model', 'value_model', 'export_data_path']:
        evaluate_config[key] += str(1)

    for i in [8032]:
        generate_config['policy_port'] = i
        policy_value_fn = PolicyValueFunction(generate_config)

    await RunMCTS_Generate(generate_config, policy_value_fn).run()

    await RunMCTS_Evaluate(evaluate_config, policy_value_fn).run()
if __name__ == "__main__":
    asyncio.run(main())
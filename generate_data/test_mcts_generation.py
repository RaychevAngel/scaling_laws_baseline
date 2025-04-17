import asyncio
import yaml
import os
import sys
from utils.policy_value import PolicyValueFunction
from generate_data.mcts_generator import RunMCTS_Generate

async def test_mcts_generation():
    # Load configuration
    with open('generate_data/config_mcts_generator.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Update paths for testing
    iteration = 1
    for key in ['policy_data_path', 'value_data_path', 'policy_model', 'value_model']:
        config[key] += str(iteration)
    
    # Set ports manually (use the same ports as your running servers)
    config['policy_port'] = 8906  # Replace with your actual policy server port
    config['value_port'] = 8064   # Replace with your actual value server port
    
    # Create policy value function
    policy_value_fn = PolicyValueFunction(config)
    # Run MCTS generation
    print("Starting MCTS generation test...")
    await RunMCTS_Generate(config, policy_value_fn.__call__).run()
    print("MCTS generation test completed.")

if __name__ == "__main__":
    asyncio.run(test_mcts_generation()) 
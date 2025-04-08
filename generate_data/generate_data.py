import asyncio
import yaml
import subprocess
import os
import re
from mcts_generator import RunMCTS_Generate

async def main(iteration: int):
    # Load configuration
    with open('config_mcts_generator.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Update iteration-specific paths
    for key in ['policy_model', 'policy_data_path', 'value_data_path']:
        config[key] += str(iteration)
    
    # Extract ports from API base URLs
    policy_port = int(re.search(r':(\d+)/', config['policy_api_base']).group(1))
    value_port = int(re.search(r':(\d+)/', config['value_api_base']).group(1))
    
    print(f"Starting servers on ports: policy={policy_port}, value={value_port}")
    
    # Start servers in new terminals
    policy_cmd = f"gnome-terminal -- python -c \"from utils.deploy_policy import deploy_policy; deploy_policy({policy_port})\""
    value_cmd = f"gnome-terminal -- python -c \"from utils.deploy_value import deploy_value; deploy_value({value_port})\""
    
    subprocess.Popen(policy_cmd, shell=True)
    subprocess.Popen(value_cmd, shell=True)
    
    # Allow time for servers to initialize
    print("Waiting 20 seconds for servers to initialize...")
    await asyncio.sleep(20)
    
    try:
        # Run the MCTS generator
        await RunMCTS_Generate(config).run()
    finally:
        # Terminate server processes
        print("Shutting down servers...")
        os.system(f"pkill -f 'deploy_policy.*{policy_port}'")
        os.system(f"pkill -f 'deploy_value.*{value_port}'")

if __name__ == "__main__":
    asyncio.run(main(iteration=1))

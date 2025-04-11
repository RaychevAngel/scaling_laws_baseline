import asyncio
import yaml
from utils.deploy import PolicyValueServer
from mcts_generator import RunMCTS_Generate

async def main(iteration: int):
    # Load configuration
    with open('config_mcts_generator.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Update iteration-specific paths
    for key in ['policy_data_path', 'value_data_path', 'policy_model', 'value_model']:
        config[key] += str(iteration)

    
    # Start the policy-value server
    print(f"Starting policy-value server on {config['host']}:{config['port']}")
    print(f"Using policy model: {config['policy_model']}")
    print(f"Using value model: {config['value_model']}")
    
    server = PolicyValueServer(
        policy_model=config['policy_model'],
        value_model=config['value_model'],
        host=config['host'],
        port=config['port'],
        endpoint=config['endpoint']
    )
    server.start()
    
    # Allow time for the server to initialize
    print("Waiting 5 seconds for server to initialize...")
    await asyncio.sleep(5)
    
    try:
        # Run the MCTS generator
        await RunMCTS_Generate(config).run()
    finally:
        # Stop the server (server will be terminated when program exits)
        server.stop()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, default=1, help="Iteration number for data generation")
    args = parser.parse_args()
    
    asyncio.run(main(iteration=args.iteration))

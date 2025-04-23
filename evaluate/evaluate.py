import asyncio
import yaml
import subprocess
import time
import os
import requests
import socket
import signal
import psutil
import random
from typing import List, Tuple
from utils.policy_value import PolicyValueFunction
from evaluate.mcts_evaluator import RunMCTS_Evaluate
from utils.env_config import get_hf_user


def find_available_port():
    """Find a random available port and update config."""
    print(f"Finding available port...")

    while True:
        port = random.randint(8000, 9000)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            port_in_use = s.connect_ex(('localhost', port)) == 0
        if not port_in_use:
            return port

def wait_for_server(host, port, endpoint, config, is_policy=False, timeout=300):
    """Wait for a server to be ready by checking if it responds to requests."""
    url = f"http://{host}:{port}{endpoint}"
    print(f"Waiting for server at {url} to be ready (timeout={timeout}s)...")

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            if socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect_ex((host, port)) == 0:
                try:
                    # Different request format for policy vs value endpoints
                    if is_policy:
                        request_data = {
                            "questions_and_states": [["Use 2,3,4,9 to make 30", ""]],
                            "branch_factor": config.get('branch_factor', 3),
                            "temperature": config.get('temperature', 1.0)
                        }
                    else:
                        request_data = {
                            "questions_and_states": [["Use 2,3,4,9 to make 30", ""]]
                        }

                    response = requests.post(url, json=request_data,
                                    headers={"Content-Type": "application/json"}, timeout=1)
                    if response.status_code < 500:
                        print(f"Server at {url} is ready!")
                        print(f"Test response: {response.json()}")
                        return True
                except Exception as e:
                    print(f"Request error: {str(e)}")
                    pass
        except:
            pass
        time.sleep(1)

    print(f"Timeout waiting for server at {url}")
    return False

async def main(iteration: int):
    # Load configuration
    with open('evaluate/config_mcts_evaluator.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Keep the source models unchanged - we're only loading from them

    # Update iteration-specific paths
    for key in ['policy_model', 'value_model', 'export_data_path']:
        config[key] += str(iteration)

    # Set CUDA devices
    policy_env = os.environ.copy()
    policy_env["CUDA_VISIBLE_DEVICES"] = "4"
    value_env = os.environ.copy()
    value_env["CUDA_VISIBLE_DEVICES"] = "5"

    # Find available ports
    config['policy_port'] = find_available_port()
    config['value_port'] = find_available_port()

    # Start policy server
    policy_cmd = ["python", "utils/deploy_policy.py", "--policy_model", config['policy_model'],
                 "--host", config['host'], "--port", str(config['policy_port']),
                 "--endpoint", config['policy_endpoint']]
    print(f"Starting policy server on {config['host']}:{config['policy_port']} (GPU 0)")
    policy_process = subprocess.Popen(policy_cmd, env=policy_env)

    # Wait for policy server
    if not wait_for_server(config['host'], config['policy_port'], config['policy_endpoint'], config, is_policy=True):
        print("Policy server failed to start. Terminating...")
        policy_process.terminate()
        return

    # Start value server
    value_cmd = ["python", "utils/deploy_value.py", "--value_model", config['value_model'],
                "--host", config['host'], "--port", str(config['value_port']),
                "--endpoint", config['value_endpoint']]
    print(f"Starting value server on {config['host']}:{config['value_port']} (GPU 1)")
    value_process = subprocess.Popen(value_cmd, env=value_env)

        # Wait for value server
    if not wait_for_server(config['host'], config['value_port'], config['value_endpoint'], config, is_policy=False):
        print("Value server failed to start. Terminating...")
        policy_process.terminate()
        value_process.terminate()
        return


    try:
        policy_value_fn = PolicyValueFunction(config)
        await RunMCTS_Evaluate(config, policy_value_fn).run()
    finally:
         # Stop servers
        print("Stopping servers...")
        policy_process.terminate()
        value_process.terminate()
        policy_process.wait()
        value_process.wait()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", type=int, default=2, help="Iteration number for evaluation")
    args = parser.parse_args()

    asyncio.run(main(iteration=args.iteration))

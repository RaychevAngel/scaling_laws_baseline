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

def find_available_port():
    """Find a random available port."""
    print(f"Finding available port...")
    
    while True:
        port = random.randint(8000, 9000)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            port_in_use = s.connect_ex(('localhost', port)) == 0
        if not port_in_use:
            return port

def wait_for_server(host, port, endpoint, is_policy=False, timeout=120):
    """Wait for a server to be ready by checking if it responds to requests."""
    url = f"http://{host}:{port}{endpoint}"
    print(f"Waiting for server at {url} to be ready...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            if socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect_ex((host, port)) == 0:
                try:
                    # Different request format for policy vs value endpoints
                    if is_policy:
                        request_data = {
                            "questions_and_states": [["Use 2,3,4,9 to make 30", ""]],
                            "branch_factor": 1,
                            "temperature": 1.0
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

async def main():
    policy_port = find_available_port()
    value_port = find_available_port()
    
    policy_cmd = ["python", "utils/deploy_policy.py", "--policy_model", "AngelRaychev/policy_iteration_1",
                 "--host", "0.0.0.0", "--port", str(policy_port), 
                 "--endpoint", "/policy-prediction"]
    value_cmd = ["python", "utils/deploy_value.py", "--value_model", "AngelRaychev/value_iteration_1",
                "--host", "0.0.0.0", "--port", str(value_port), 
                "--endpoint", "/value-prediction"]
    
    policy_env = os.environ.copy()
    policy_env["CUDA_VISIBLE_DEVICES"] = "0"
    value_env = os.environ.copy()
    value_env["CUDA_VISIBLE_DEVICES"] = "1"

    policy_process = subprocess.Popen(policy_cmd, env=policy_env)
    if not wait_for_server("0.0.0.0", policy_port, "/policy-prediction", is_policy=True):
        print("Policy server failed to start. Terminating...")
        policy_process.terminate()
        return
    
    value_process = subprocess.Popen(value_cmd, env=value_env)
    if not wait_for_server("0.0.0.0", value_port, "/value-prediction", is_policy=False):
        print("Value server failed to start. Terminating...")
        policy_process.terminate()
        value_process.terminate()
        return

    qs = [("Use 2, 2, 5, 10 to make 1.", ""),
          ("Use 2, 2, 5, 10 to make 1.", "10-5=5 (left: 2, 2, 5)\n"),
          ("Use 2, 2, 5, 10 to make 1.", "10-5=5 (left: 2, 2, 5)\n2*2=4 (left: 4, 5)\n"),
          ("Use 2, 2, 5, 10 to make 1.", "10-5=5 (left: 2, 2, 5)\n2*2=4 (left: 4, 5)\n5-4=1 (left: 1)\n"),
          ("Use 2, 2, 5, 10 to make 1.", "10-5=5 (left: 2, 2, 5)\n2*2=4 (left: 4, 5)\n5-4=1 (left: 1)\nThe answer is:")
          ("Use 2, 2, 5, 10 to make 1.", "10-5=5 (left: 2, 2, 5)\n2*2=4 (left: 4, 5)\n5-4=1 (left: 1)\nThe answer is: 10-5-2*2=1.\n")
          ]


    policy_resp = requests.post(
                url=f"http://0.0.0.0:{policy_port}/policy-prediction",
                json={
                    "questions_and_states": qs, 
                    "branch_factor": 1,
                    "temperature": 1.0
                },
                headers={"Content-Type": "application/json"},
                timeout=60
            )
    value_resp = requests.post(
                url=f"http://0.0.0.0:{value_port}/value-prediction",
                json={"questions_and_states": qs},
                headers={"Content-Type": "application/json"},
                timeout=60
            )
    print(policy_resp.json()['results'])
    print(value_resp.json()['results'])

if __name__ == "__main__":
    asyncio.run(main())
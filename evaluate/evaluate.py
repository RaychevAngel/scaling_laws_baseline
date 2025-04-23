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
from typing import List, Tuple, Dict
from utils.policy_value import PolicyValueFunction
from evaluate.mcts_evaluator import RunMCTS_Evaluate


class Evaluator:
    def __init__(self, config: Dict, policy_gpu: int, value_gpu: int):
        self.policy_gpu = policy_gpu
        self.value_gpu = value_gpu
        self.policy_process = None
        self.value_process = None
        self.config = config

    def _find_available_port(self):
        """Find a random available port."""
        print(f"Finding available port...")
        
        while True:
            port = random.randint(8000, 9000)
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(('localhost', port)) != 0:
                    return port

    def _wait_for_server(self, server_type: str, host: str, port: int, endpoint: str, is_policy: bool, timeout: int = 300):
        """Wait for a server to be ready by checking if it responds to requests."""
        url = f"http://{host}:{port}{endpoint}"
        print(f"Waiting for {server_type} server at {url} to be ready (timeout={timeout}s)...")
        
        start_time = time.time()
        last_error = ""
        
        while time.time() - start_time < timeout:
            try:
                with socket.create_connection((host, port), timeout=0.5):
                    # Different request format for policy vs value endpoints
                    if is_policy:
                        request_data = {
                            "questions_and_states": [["Use 2,3,4,9 to make 30", ""]],
                            "branch_factor": self.config.get('branch_factor', 3),
                            "temperature": self.config.get('temperature', 1.0)
                        }
                    else:
                        request_data = {
                            "questions_and_states": [["Use 2,3,4,9 to make 30", ""]]
                        }
                    
                    response = requests.post(url, json=request_data, 
                                    headers={"Content-Type": "application/json"}, timeout=2)
                    if response.status_code < 500:
                        print(f"Server at {url} is ready!")
                        print(f"Test response: {response.json()}")
                        return True
                    else:
                        last_error = f"Server responded with status {response.status_code}"
            except requests.exceptions.RequestException as e:
                last_error = f"Request failed: {e}"
            except Exception as e:
                last_error = f"Connection/request error: {e}"
            
            time.sleep(1)
        
        print(f"ERROR: Timeout waiting for {server_type} server. Last error: {last_error}")
        return False

    def _start_server(self, server_type: str, script_path: str, model_key: str, port_key: str, endpoint_key: str, gpu_id: int) -> bool:
        """Start a policy or value server on the specified GPU."""
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        required_keys = ['host', model_key, endpoint_key]
        if not all(key in self.config for key in required_keys):
            print(f"ERROR: Missing required config keys for {server_type} server")
            return False
            
        port = self._find_available_port()
        self.config[port_key] = port
        host = self.config['host']
        model = self.config[model_key]
        endpoint = self.config[endpoint_key]
        
        cmd = ["python", script_path, 
               f"--{model_key}", model,
               "--host", host, 
               "--port", str(port), 
               "--endpoint", endpoint]
        
        print(f"Starting {server_type} server on {host}:{port} (GPU {gpu_id})")
        try:
            process = subprocess.Popen(cmd, env=env)
        except FileNotFoundError:
            print(f"ERROR: Script not found: {script_path}")
            return False
        except Exception as e:
            print(f"ERROR: Failed to start {server_type} server: {e}")
            return False
            
        setattr(self, f"{server_type}_process", process)

        is_policy = (server_type == "policy")
        if not self._wait_for_server(server_type, host, port, endpoint, is_policy=is_policy):
            print(f"Terminating failed {server_type} server...")
            process.kill()
            process.wait()
            setattr(self, f"{server_type}_process", None)
            return False
            
        return True

    async def run(self):
        """Run the evaluation process."""
        try:
            print("--- Starting Servers ---")
            if not self._start_server("policy", "utils/deploy_policy.py", "policy_model", "policy_port", "policy_endpoint", self.policy_gpu):
                print("Aborting: policy server failed")
                return
                
            if not self._start_server("value", "utils/deploy_value.py", "value_model", "value_port", "value_endpoint", self.value_gpu):
                print("Aborting: value server failed")
                return
            
            print("--- Running MCTS Evaluation ---")
            policy_value_fn = PolicyValueFunction(self.config)
            mcts_runner = RunMCTS_Evaluate(self.config, policy_value_fn)
            await mcts_runner.run()
            print("--- MCTS Evaluation Complete ---")
            
        except AttributeError as e:
            print(f"ERROR: Missing config key: {e}")
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("--- Stopping Servers ---")
            # Forcefully kill any remaining server processes
            try:
                subprocess.run(["pkill", "-9", "-f", "deploy_policy.py"])
                subprocess.run(["pkill", "-9", "-f", "deploy_value.py"])
            except Exception as e:
                print(f"Error during forceful process cleanup: {e}")

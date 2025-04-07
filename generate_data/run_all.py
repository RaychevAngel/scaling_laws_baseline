import multiprocessing
import subprocess
import sys
import time
import signal
import os

def run_policy_server():
    """Run the policy server on port 9876"""
    subprocess.run([sys.executable, "deploy/deploy_policy.py"], check=True)

def run_value_server():
    """Run the value server on port 8000"""
    subprocess.run([sys.executable, "deploy/deploy_value.py"], check=True)

def run_mcts():
    """Run the MCTS process"""
    subprocess.run([sys.executable, "mcts/mcts.py"], check=True)

def signal_handler(signum, frame):
    """Handle termination signals"""
    print("\nReceived termination signal. Shutting down...")
    sys.exit(0)

def main():
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Create processes
    processes = []
    
    try:
        # Start policy server
        policy_process = multiprocessing.Process(target=run_policy_server)
        policy_process.start()
        processes.append(policy_process)
        print("Started policy server...")
        
        # Wait a bit to ensure policy server is up
        time.sleep(2)
        
        # Start value server
        value_process = multiprocessing.Process(target=run_value_server)
        value_process.start()
        processes.append(value_process)
        print("Started value server...")
        
        # Wait a bit to ensure value server is up
        time.sleep(2)
        
        # Start MCTS
        mcts_process = multiprocessing.Process(target=run_mcts)
        mcts_process.start()
        processes.append(mcts_process)
        print("Started MCTS process...")
        
        # Wait for all processes
        for process in processes:
            process.join()
            
    except KeyboardInterrupt:
        print("\nShutting down processes...")
        for process in processes:
            process.terminate()
            process.join()
    except Exception as e:
        print(f"Error: {e}")
        for process in processes:
            process.terminate()
            process.join()
        sys.exit(1)

if __name__ == "__main__":
    main() 
from utils.deploy_policy import PolicyServer
import time
import signal
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Skip GPU 0, make GPUs 1-7 available if they exist

# Handle graceful shutdown with Ctrl+C
def signal_handler(sig, frame):
    print('Shutting down server...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

policy_server = PolicyServer(
    policy_model="AngelRaychev/0.5B-policy-iteration_1",
    host="0.0.0.0",
    port=8032,
    endpoint="/policy-prediction",)
#    revision="5d9ff93fed23a4189948cb7a6e1c1ea40c43e865")

# Start the server in background thread
policy_server.start()

print("Server running. Press CTRL+C to stop.")

# Keep the main thread alive to prevent the program from exiting
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Shutting down...")
import os
from utils.deploy_value import ValueServer
import time
import signal
import sys

# Hide GPU 0 from the process, but make other GPUs available if they exist
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # Skip GPU 0, make GPUs 1-7 available if they exist

# Handle graceful shutdown with Ctrl+C
def signal_handler(sig, frame):
    print('Shutting down server...')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

value_server = ValueServer(
    value_model="AngelRaychev/0.5B-value-iteration_6",
    host="127.0.0.1",
    port=8056,
    endpoint="/value-prediction",)
#    revision="7a9818b7d779398639928aafc4a09dbf5e3e8324")

# Start the server in background thread
value_server.start()

print("Server running. Press CTRL+C to stop.")

# Keep the main thread alive to prevent the program from exiting
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Shutting down...")
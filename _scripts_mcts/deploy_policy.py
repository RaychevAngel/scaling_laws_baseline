import os
from utils.deployer_policy import PolicyServer
import time
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--iter", type=int, required=True)
parser.add_argument("--gpu", type=int, required=True)
parser.add_argument("--port", type=int, required=True)
args = parser.parse_args()

########################################################
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
policy_model_checkpoint = args.iter
policy_model_port = 8050 + 4*args.gpu + 2*args.port
########################################################

policy_server = PolicyServer(
    policy_model="AngelRaychev/1.5B-policy-iteration_" + str(policy_model_checkpoint),
    revision=None,
    host="127.0.0.1",
    port=policy_model_port,
    endpoint="/policy-prediction",
    gpu_memory_utilization=0.45
    )

policy_server.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Shutting down...")
    policy_server.stop()
    sys.exit(0)
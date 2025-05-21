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
sos_model_checkpoint = args.iter
sos_model_port = 8050 + 4*args.gpu + 2*args.port
########################################################

sos_server = PolicyServer(
    policy_model="AngelRaychev/0.5B-sos-iteration_" + str(sos_model_checkpoint),
    revision=None,
    host="127.0.0.1",
    port=sos_model_port,
    endpoint="/policy-prediction",
    gpu_memory_utilization=0.22
    )

sos_server.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Shutting down...")
    sos_server.stop()
    sys.exit(0)
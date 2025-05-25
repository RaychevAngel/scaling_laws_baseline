import os
from utils.deployer_value import ValueServer
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
value_model_checkpoint = args.iter
value_model_port = 8050 + 4*args.gpu + 2*args.port + 1
########################################################

value_server = ValueServer(
    value_model="AngelRaychev/0.5B-value-iteration_" + str(value_model_checkpoint),
    revision=None,
    host="127.0.0.1",
    port=value_model_port,
    endpoint="/value-prediction",
    gpu_memory_utilization=0.23
    )

value_server.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Shutting down...")
    value_server.stop()
    sys.exit(0)
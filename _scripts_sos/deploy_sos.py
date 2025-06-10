import os
from utils.deployer_sos import SosServer
import time
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--iter", type=int, required=True)
parser.add_argument("--gpu", type=int, required=True)
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--b", type=int, required=False)
parser.add_argument("--e", type=int, required=False)
parser.add_argument("--epochs", type=int, required=False)
parser.add_argument("--tokens", type=int, required=False)
args = parser.parse_args()

########################################################
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
sos_model_checkpoint = args.iter
sos_model_port = 8050 + 4*args.gpu + args.port
be_extension = f"_b{args.b}_e{args.e}_epochs{args.epochs}"
########################################################

sos_server = SosServer(
    sos_model="AngelRaychev/3B-sos-iteration_" + str(sos_model_checkpoint) + be_extension,
    revision=None,
    host="127.0.0.1",
    port=sos_model_port,
    endpoint="/sos-prediction",
    gpu_memory_utilization=0.95,
    max_tokens=args.tokens
    )

sos_server.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Shutting down...")
    sos_server.stop()
    sys.exit(0)
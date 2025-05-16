import os
from utils.deployer_policy import PolicyServer
import time
import sys

########################################################
i = 7
k = 13
########################################################
iteration = i 
policy_port = 8050+2*k
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
########################################################

policy_server = PolicyServer(
    policy_model="AngelRaychev/0.5B-policy-iteration_" + str(iteration),
    host="127.0.0.1",
    port=policy_port,
    endpoint="/policy-prediction",
    revision=None
    )

policy_server.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Shutting down...")
    policy_server.stop()
    sys.exit(0)
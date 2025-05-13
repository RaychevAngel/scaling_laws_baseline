import os
from utils.deployer_policy import PolicyServer
import time
import sys

########################################################
i = 1
j = 6
policy_port = 8050+j
os.environ["CUDA_VISIBLE_DEVICES"] = str(j)
########################################################

policy_server = PolicyServer(
    policy_model="AngelRaychev/0.5B-policy-iteration_" + str(i),
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
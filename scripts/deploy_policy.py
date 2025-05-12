import os
from utils.deployer_policy import PolicyServer
import time
import sys

########################################################
i = 3
policy_port = 8052
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
########################################################

policy_server = PolicyServer(
    policy_model="AngelRaychev/0.5B-policy-iteration_" + str(i),
    host="127.0.0.1",
    port=policy_port,
    endpoint="/policy-prediction"
    )

policy_server.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Shutting down...")
    policy_server.stop()
    sys.exit(0)
import os
from utils.deployer_value import ValueServer
import time
import sys

########################################################
i = 3
k = 7
value_port = 8050+2*k + 1
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
########################################################
value_server = ValueServer(
    value_model="AngelRaychev/0.5B-value-iteration_" + str(i),
    host="127.0.0.1",
    port=value_port,
    endpoint="/value-prediction",
    revision=None
    )


value_server.start()

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Shutting down...")
    value_server.stop()
    sys.exit(0)
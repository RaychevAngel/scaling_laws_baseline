import os
from utils.deployer_value import ValueServer
import time
import sys

########################################################
i = 1
j = 7
value_port = 8050+j
os.environ["CUDA_VISIBLE_DEVICES"] = str(j)
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
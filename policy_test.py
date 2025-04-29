import os
import requests
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

url = "http://127.0.0.1:8050/policy-prediction"

q = "Use 7, 7, 8, 12 to make 61.\n"
s1 = "7*8=56 (left: 7, 12, 56)\n"
s2 = "12-7=5 (left: 5, 56)\n"
s3 = "56+5=61 (left: 61)\n"
a = "The answer is: (7*8)+(12-7)= 61.\n"
inputs = [[q,""], [q, s1], [q, s1 + s2], [q, s1 + s2 + s3], [q, s1 + s2 + s3 + a]]
response = requests.post(url, json={"questions_and_states": inputs, "branch_factor": 1, "temperature": 0.0})
for i, r in enumerate(response.json()["results"]):
    print(f"Input {i+1}: {inputs[i]}")
    print(f"Output {i+1}: {r}")
    print()
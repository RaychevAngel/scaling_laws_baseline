import os
import requests
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

url = "http://127.0.0.1:8051/value-prediction"
batch_size = 256

q = "Use 7, 7, 8, 12 to make 61.\n"
s1 = "7+8=15 (left: 7, 12, 15)\n"
s2 = "12-7=5 (left: 5, 15)\n"
s3 = "15/5=3 (left: 3)\n"
a = "The answer is: (7+8)/(12-7)= 3.\n"
inputs = [[q,""], [q, s1], [q, s1 + s2], [q, s1 + s2 + s3], [q, s1 + s2 + s3 + a]]
response = requests.post(url, json={"questions_and_states": inputs})
for r in response.json():
    print(r)
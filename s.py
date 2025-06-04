import os
import requests

qs = [("Use 2, 5, 10, 11 to make 1.\n", "10+11=21 Left: 2, 5, 21\n5-2=3 Left: 3, 21\n21/3=7 Left: 7\n")]

for q in qs:
    resp = requests.post(
        url="http://127.0.0.1:8050/policy-prediction",
        json={"questions_and_states": [q], "branch_factor": 10, "temperature": 1.0},
        headers={"Content-Type": "application/json"},
        timeout=60
    )
    for r in resp.json()['results']:
        print(r)
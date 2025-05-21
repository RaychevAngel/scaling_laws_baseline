import requests

current="""Use 1, 4, 9, 11 to make 25.
11*2=22 (left: 1, 4, 2
4*11=44 (left: 1, 9, 44)
22+4=26 (left: 1, 26)
44-9=35 (left: 1, 35)
26+1=27 (left: 27)
44-9=35 (left: 1, 35)
22+1=23 (left: 4, 23)
44-35=9 (left: 9)
26+1=27 (left: 27)
27+2=29 (left: 29)
44-1=43 (left: 9, 43)
1+4=5 (left: 5, 2)
35-1=34 (left: 34)
4+1=5 (left: 5, 2)
2*5=10 (left: 10, 44)
The answer is: 4*11-9+2= 29.
23+4=27 (left: 27)
35-1=34 (left: 34)
23+4=27 (left: 27)
The answer is: 4*11-9+1= 27.
35-1=34 (left: 34)
34-5=29 (left: 29)
10+4=14 (left: 14)
35-1=34 (left: 34)
The answer is: 4*11-9+1= 29.
The answer is: 4*11-9+1= 29.
Final answer:"""

for i  in range(100):
    qs = [(current,"")]

    url = "http://127.0.0.1:8050/policy-prediction"
    sos_resp = requests.post(
        url=url,
        json={
            "questions_and_states": qs,
            "branch_factor": 1,
            "temperature": 1.0
        },
        headers={"Content-Type": "application/json"},
        timeout=60
    )

    sos_result = sos_resp.json()['results'][0][0]
    print(sos_result)


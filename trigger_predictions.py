# trigger_predictions.py
import requests
import random
import time

URL = "http://localhost:8000/predict"

for i in range(15):
    if i % 3 == 0:
        landmarks = [random.uniform(0, 1) for _ in range(10)]
    else:
        landmarks = [random.uniform(0, 1) for _ in range(63)]

    payload = {"landmarks": landmarks}

    try:
        res = requests.post(URL, json=payload)
        print(f"Request {i+1}: {res.status_code} - {res.json()}")
    except Exception as e:
        print(f"Request {i+1}: Failed - {e}")

    time.sleep(1)
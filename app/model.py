import joblib
import numpy as np
import json

model = joblib.load("RF_model.pkl")
with open("label_mapping.json", "r") as f:
    label_mapping = json.load(f)
inv_label_mapping = {v: k for k, v in label_mapping.items()}

def predict(landmarks_flat: list):
    arr = np.array(landmarks_flat).reshape(1, -1)
    pred = model.predict(arr)[0]
    label = inv_label_mapping[pred]
    return label

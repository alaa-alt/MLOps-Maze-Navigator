from fastapi import FastAPI
from app.schema import Landmarks
from app.model import predict

app = FastAPI(title="Hand Gesture Classifier API")

@app.get("/")
def root():
    return {"message": "API is up and running!"}

@app.post("/predict")
def classify_hand(landmarks: Landmarks):
    prediction = predict(landmarks.landmarks)
    return {"predicted_label": prediction}
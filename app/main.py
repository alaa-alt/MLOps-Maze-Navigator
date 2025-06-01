from fastapi import FastAPI, HTTPException
from app.schema import Landmarks
from app.model import predict

app = FastAPI(title="Hand Gesture Classifier API")

@app.get("/")
def root():
    return {"message": "API is up and running!"}

@app.post("/predict")
def classify_hand(landmarks: Landmarks):
    try:
        prediction = predict(landmarks.landmarks)
        return {"predicted_label": prediction}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
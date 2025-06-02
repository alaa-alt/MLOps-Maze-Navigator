import numpy as np
from fastapi import FastAPI, HTTPException, Request, Response
from schema import Landmarks
from model import predict
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from prometheus_fastapi_instrumentator import Instrumentator
import time
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI(title="Hand Gesture Classifier API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or replace * with your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
Instrumentator().instrument(app).expose(app)

REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests",
    ["method", "endpoint", "http_status"]
)
INVALID_INPUT_COUNT = Counter(
    "invalid_input_total", "Total invalid hand landmark inputs"
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds", "Time taken to make predictions"
)

@app.get("/")
def root():
    return {"message": "API is up and running!"}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
@PREDICTION_LATENCY.time()
def classify_hand(landmarks: Landmarks, request: Request):
    try:
        landmarks = landmarks.landmarks
        landmarks_3d = [tuple(landmarks[i:i+3]) for i in range(0, len(landmarks), 3)]

        # Normalize input
        wrist = np.array(landmarks_3d[0][:2])           # x1, y1
        middle_tip = np.array(landmarks_3d[12][:2])     # x13, y13
        scale = np.linalg.norm(wrist - middle_tip) or 1

        normalized = [
            ((x - wrist[0]) / scale, (y - wrist[1]) / scale, z / scale)
            for x, y, z in landmarks_3d
        ]

        # Flatten back to a 63-element list
        normalized_flat = [coord for point in normalized for coord in point]

        prediction = predict(normalized_flat)
        REQUEST_COUNT.labels(request.method, request.url.path, "200").inc()
        print(prediction)
        return {"predicted_label": prediction}
    except ValueError as e:
        INVALID_INPUT_COUNT.inc()
        REQUEST_COUNT.labels(request.method, request.url.path, "422").inc()
        raise HTTPException(status_code=422, detail=str(e))

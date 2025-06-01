from fastapi import FastAPI, HTTPException, Request, Response
from schema import Landmarks
from model import predict
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from prometheus_fastapi_instrumentator import Instrumentator
import time

app = FastAPI(title="Hand Gesture Classifier API")

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
        prediction = predict(landmarks.landmarks)
        REQUEST_COUNT.labels(request.method, request.url.path, "200").inc()
        return {"predicted_label": prediction}
    except ValueError as e:
        INVALID_INPUT_COUNT.inc()
        REQUEST_COUNT.labels(request.method, request.url.path, "422").inc()
        raise HTTPException(status_code=422, detail=str(e))

# Hand Gesture Maze Navigation – Backend API

This repository contains the backend system for hand gesture recognition used to control a maze navigation game. The backend is developed using **FastAPI**, containerized with **Docker**, and monitored using **Prometheus** and **Grafana**.

---

## Model Serving

- Framework: FastAPI  
- Main API logic is implemented in `main.py`  
- Endpoint `/predict` accepts hand landmarks and returns a direction label  
- Code is modular and includes separation for:
  - `model.py`: loads the gesture classification model
  - `schema.py`: defines request/response models
  - `trigger_predictions.py`: tester function for /predict

---

## Unit Testing

- Unit tests are written using `pytest`
- Tests are located in `test_main.py`
- To run tests:
  ```bash
  pytest test_main.py
  ```

---

## Containerization

- The application is containerized using a `Dockerfile`
- To build and run the container:
  ```bash
  docker build -t hand-gesture-api .
  docker run -p 8000:8000 hand-gesture-api
  ```

---

## Monitoring Metrics

The system collects and exposes the following Prometheus metrics:

| Category         | Metric Name                | Description                                                               |
|------------------|----------------------------|---------------------------------------------------------------------------|
| Model-related     | `prediction_latency_seconds` | Measures response time for prediction requests                            |
| Data-related      | `invalid_inputs_total`       | Counts malformed or invalid landmark inputs                               |
| Server-related    | `app_memory_usage_bytes`     | Tracks memory usage of the application container                          |

Metrics are defined and exposed in the FastAPI application, scraped by Prometheus, and visualized with Grafana.

---

## System Monitoring

- Monitoring stack includes:
  - Prometheus (`prometheus.yml`)
  - Grafana (configured via `docker-compose.yml`)
- Metrics are visualized on a Grafana dashboard that includes:
  - App Memory Usage
  - Average Prediction Latency
  - Count of Invalid Prediction Inputs

---

## How to Run Locally

```bash
git clone <repository-url>
cd <repository>
docker-compose up --build
```

Access the services at:
- FastAPI Docs: [http://localhost:8000/docs](http://localhost:8000/docs)  
- Grafana: [http://localhost:3000](http://localhost:3000) (default: `admin` / `admin`)

---

## Project Structure

```
.
├── docker-compose.yml         # Service orchestration
├── dockerfile                 # App container definition
├── label_mapping.json         # Gesture label mapping
├── main.py                    # FastAPI app entry point
├── model.py                   # Loads and predicts with RF model
├── prometheus.yml             # Prometheus scrape configuration
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
├── RF_model.pkl               # Trained gesture recognition model
├── schema.py                  # Request/response data schemas
├── test_main.py               # Unit tests
├── trigger_predictions.py     # Helper function to test/predict
```

---

## Dependencies

Install dependencies locally (optional):
```bash
pip install -r requirements.txt
```

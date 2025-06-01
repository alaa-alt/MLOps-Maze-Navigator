# MLOps-Maze-Navigator
## Monitoring Overview

As part of the MLOps pipeline for our FastAPI-based hand gesture prediction service, we implemented a real-time monitoring dashboard using Prometheus and Grafana. The dashboard includes three carefully chosen metrics that offer full visibility across the model, data, and infrastructure layers.

---

### 1. Model-Related: `prediction_latency_seconds`

**What it shows:**  
The average time (in seconds) taken by the model to produce a prediction.

**Why we chose it:**  
Inference latency is a core performance indicator for any deployed ML model. Monitoring this metric allows us to:
- Detect model slowness or performance degradation
- Track how the model scales with load
- Ensure responsiveness remains within acceptable limits for user experience

**Prometheus query used:**
```prometheus
rate(prediction_latency_seconds_sum[1m]) / rate(prediction_latency_seconds_count[1m])
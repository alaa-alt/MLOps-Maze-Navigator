# MLOps-Maze-Navigator
## Monitoring Metrics

We use Prometheus to monitor the application's health, performance, and data quality. The selected metrics and their justifications are:

- **Model-Related**: `prediction_latency_seconds`
  - Measures the time taken by the model to return predictions.
  - Useful for detecting performance bottlenecks in real-time inference.

- **Data-Related**: `invalid_input_total`
  - Counts how many invalid or malformed inputs are received by the API.
  - Helps detect data quality issues from the frontend or other clients.

- **Server-Related**: `http_requests_total`
  - Tracks the number of HTTP requests categorized by response status code (e.g., 200, 400, 500).
  - Gives insight into overall server stability and error rates.


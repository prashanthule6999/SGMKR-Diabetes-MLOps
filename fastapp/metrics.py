# Prometheus metrics
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info
)

# ------------------------------------------------------------------------------
# Request Metrics
# ------------------------------------------------------------------------------

PREDICTION_REQUESTS = Counter(
    "prediction_requests_total",
    "Total prediction requests"
)

PREDICTION_SUCCESS = Counter(
    "prediction_success_total",
    "Successful prediction requests"
)

PREDICTION_FAILURE = Counter(
    "prediction_failure_total",
    "Failed prediction requests"
)

# ------------------------------------------------------------------------------
# Prediction Distribution
# ------------------------------------------------------------------------------

PREDICTION_RESULT = Counter(
    "prediction_result_total",
    "Prediction distribution",
    ["result"]
)

# ------------------------------------------------------------------------------
# Latency
# ------------------------------------------------------------------------------

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds"
)

# ------------------------------------------------------------------------------
# Model Metrics
# ------------------------------------------------------------------------------

ENDPOINT_AVAILABLE = Gauge(
    "sagemaker_endpoint_available",
    "Whether the SageMaker endpoint is available"
)

ENDPOINT_INFO = Info(
    "sagemaker_endpoint",
    "SageMaker endpoint information"
)
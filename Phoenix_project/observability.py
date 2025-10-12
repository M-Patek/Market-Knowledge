# observability.py

import logging
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# --- Metric Definitions ---

# DataManager Cache Metrics
CACHE_HITS = Counter(
    "phoenix_cache_hits_total",
    "Total number of data cache hits."
)
CACHE_MISSES = Counter(
    "phoenix_cache_misses_total",
    "Total number of data cache misses."
)

# AI Client Metrics
AI_CALL_LATENCY = Histogram(
    "phoenix_ai_call_latency_seconds",
    "Latency of calls to the AI API, in seconds.",
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 90.0, float("inf")],
    labels=['client']
)

# Data Provider Metrics
PROVIDER_REQUESTS_TOTAL = Counter(
    "phoenix_provider_requests_total",
    "Total number of requests to a data provider.",
    ["provider"]
)
PROVIDER_ERRORS_TOTAL = Counter(
    "phoenix_provider_errors_total",
    "Total number of errors from a data provider.",
    ["provider"]
)
PROVIDER_LATENCY_SECONDS = Histogram(
    "phoenix_provider_latency_seconds",
    "Latency of requests to a data provider.",
    ["provider"]
)
PROVIDER_DATA_FRESHNESS_SECONDS = Gauge(
    "phoenix_provider_data_freshness_seconds",
    "The difference in seconds between when data was observed and its latest available timestamp.",
    ["provider"]
)

# --- [NEW] Probabilistic Reasoning Metrics ---
PROBABILITY_CALIBRATION_BRIER_SCORE = Gauge(
    "phoenix_probability_calibration_brier_score",
    "The Brier score loss, measuring the accuracy of probabilistic predictions."
)
# Backtest Execution Metrics
BACKTEST_DURATION = Gauge(
    "phoenix_backtest_duration_seconds",
    "Duration of the last completed backtest run, in seconds."
)
TRADES_EXECUTED = Counter(
    "phoenix_trades_executed_total",
    "Total number of trades executed by the strategy."
)

# --- Server Function ---

def start_metrics_server(port: int = 8000):
    """Starts the Prometheus metrics HTTP server in a daemon thread."""
    try:
        start_http_server(port)
        logging.getLogger("PhoenixProject.Observability").info(
            f"Prometheus metrics server started on http://localhost:{port}"
        )
    except Exception as e:
        logging.getLogger("PhoenixProject.Observability").error(
            f"Failed to start Prometheus metrics server: {e}"
        )

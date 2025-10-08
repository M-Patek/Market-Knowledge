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
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 90.0, float("inf")]
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

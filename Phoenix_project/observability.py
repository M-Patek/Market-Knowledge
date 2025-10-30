import logging
import sys
import uuid

# (L7) Task 2: Import Prometheus client
from prometheus_client import Counter, Histogram, start_http_server
import time


class TraceFormatter(logging.Formatter):
    """
    (L7 Task 1) A custom formatter to include trace_id and stage_name.
    """
    def format(self, record):
        # Set default values if not provided in 'extra'
        if not hasattr(record, 'trace_id'):
            record.trace_id = 'N/A'
        if not hasattr(record, 'stage_name'):
            record.stage_name = 'N/A'
        
        # (L7 Task 1) Add trace_id and stage_name to the log format
        self._style._fmt = '%(asctime)s - [%(trace_id)s] - [%(stage_name)s] - %(name)s - %(levelname)s - %(message)s'
        return super().format(record)


def get_logger(name, level=logging.INFO):
    """
    (L7 Task 1) Creates a standardized logger with trace_id/stage_name context.
    """
    logger = logging.getLogger(name)
    if not logger.handlers: # Avoid duplicate handlers
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        formatter = TraceFormatter() # (L7 Task 1) Use our custom formatter
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


# Get the monitoring-specific logger
metric_logger = get_logger("PhoenixProject.Metrics")

# --- (L7 Task 1) Trace ID Generation ---
def generate_trace_id():
    """Generates a unique trace ID."""
    return f"trace-{uuid.uuid4().hex[:12]}"
    
# --- (L7 Task 2) Prometheus Metrics Definitions ---

AGENT_LATENCY = Histogram(
    'agent_latency_seconds',
    'Latency of L1 agent calls',
    ['agent_name']
)

FUSION_CONFLICTS = Counter(
    'fusion_conflict_count',
    'Total number of conflicts detected by the L2 fusion engine'
)

UNCERTAINTY_DISTRIBUTION = Histogram(
    'uncertainty_distribution',
    'Distribution of L3 final uncertainty scores',
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)

def start_metrics_server(port: int = 8000):
    """Starts the Prometheus metrics server."""
    start_http_server(port)
    logger.info(f"Prometheus metrics server started on port {port}")


# --- (L7 Patched) Monitoring Hooks ---

def log_agent_latency(agent_name: str, latency_sec: float):
    """Captures pipeline execution latency."""
    AGENT_LATENCY.labels(agent_name=agent_name).observe(latency_sec)

def log_l2_uncertainty(score: float):
    """Captures the L2 cognitive uncertainty score."""
    UNCERTAINTY_DISTRIBUTION.observe(score)

def log_fusion_conflict(count: int = 1):
    """Increments the fusion conflict counter."""
    FUSION_CONFLICTS.inc(count)

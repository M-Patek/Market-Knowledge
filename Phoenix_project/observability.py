import logging
import sys

def get_logger(name, level=logging.INFO):
    """
    Creates a standardized logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers: # Avoid duplicate handlers
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


# Get the monitoring-specific logger
metric_logger = get_logger("PhoenixProject.Metrics")

# --- Task 5.3: Comprehensive Monitoring Hooks ---

def log_api_call(agent_name: str, token_usage: int):
    """Captures API call count and token usage (segmented by Agent)."""
    # In a real system, this would increment a Prometheus counter
    metric_logger.info(f"MONITORING_METRIC: api_call_count=1, agent={agent_name}")
    metric_logger.info(f"MONITORING_METRIC: api_token_usage={token_usage}, agent={agent_name}")

def log_pipeline_latency(pipeline_name: str, latency_sec: float):
    """Captures pipeline execution latency."""
    # In a real system, this would update a Prometheus histogram
    metric_logger.info(f"MONITORING_METRIC: pipeline_execution_latency_seconds={latency_sec}, pipeline={pipeline_name}")

def log_l2_uncertainty(score: float):
    """Captures the L2 cognitive uncertainty score."""
    # In a real system, this would set a Prometheus gauge
    metric_logger.info(f"MONITORING_METRIC: l2_cognitive_uncertainty_score={score}")

def log_l3_rules_generated(count: int):
    """Captures the count of new L3 rules."""
    # In a real system, this would increment a Prometheus counter
    metric_logger.info(f"MONITORING_METRIC: l3_rules_generated_count={count}")

# Phoenix_project/monitor/metrics.py
# (Recreated from imports in other modules)
from prometheus_client import Gauge, Histogram

# --- L1 ---
L1_LAT = Histogram(
    'phoenix_l1_agent_latency_seconds', 
    'Latency of individual L1 agents', 
    ['agent_name']
)

# --- L2 (Evaluation) ---
UNCERTAINTY = Histogram(
    'phoenix_l2_uncertainty_score', 
    'Distribution of final uncertainty scores from fusion/voter'
)
CONSISTENCY = Histogram(
    'phoenix_l2_consistency_score', 
    'Distribution of L1 agent consistency scores'
)

# --- Calibration ---
PROBABILITY_CALIBRATION_BRIER_SCORE = Gauge(
    'phoenix_calibration_brier_score', 
    'Brier Score Loss for the probability calibration model (legacy)'
)

# --- L3 ---
L3_RULE_TRIGGERS = Gauge(
    'phoenix_l3_rules_triggered_total', 
    'Number of L3 rules triggered in the last run'
)

# --- System ---
CIRCUIT_BREAKER_STATE = Gauge(
    'phoenix_circuit_breaker_state', 
    'State of the circuit breaker (1=Open, 0=Closed)'
)

# Phoenix_project/observability/metrics.py
from prometheus_client import Counter, Histogram

REQ_TOTAL = Counter("http_requests_total", "Total HTTP requests", ["path"])
L1_LAT = Histogram("l1_agent_latency_seconds", "Latency of L1 agents")
FUSION_CONFLICTS = Counter("fusion_conflict_count", "Number of L2 conflicts resolved")
UNCERTAINTY = Histogram("uncertainty_score", "Distribution of uncertainty scores")

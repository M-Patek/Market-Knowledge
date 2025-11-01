from prometheus_client import start_http_server, Counter, Gauge, Histogram
import time
from typing import Dict, Any

from .monitor.logging import get_logger

logger = get_logger(__name__)

class MetricsCollector:
    """
    A centralized service for managing and exposing system metrics
    via a Prometheus endpoint.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        config:
          - port: int (Prometheus exposition port), default: 9108
        """
        self.port = int(config.get("port", 9108))

        # Counters
        self.orders_total = Counter(
            'phoenix_orders_total',
            'Total number of orders processed',
            ['status']  # e.g., "filled", "rejected", "pending"
        )

        self.events_processed = Counter(
            'phoenix_events_total',
            'Total number of market/text events processed',
            ['type']  # e.g., "market_tick", "news_article"
        )

        # Gauges
        self.position_value = Gauge(
            'phoenix_position_value',
            'Total marked to market value of positions'
        )

        self.cognitive_uncertainty = Gauge(
            'phoenix_cognitive_uncertainty_percent',
            'Cognitive uncertainty of the last AI decision',
            ['decision']
        )

        self.circuit_breaker_state = Gauge(
            'phoenix_circuit_breaker_state',
            'State of the main circuit breaker (0=CLOSED, 1=OPEN, 2=HALF_OPEN)'
        )

        # Histogram (Tracks distribution of values)
        self.order_latency = Histogram(
            'phoenix_order_latency_seconds',
            'Latency for order submission to acknowledgment'
        )

    def start(self):
        """Starts the Prometheus HTTP server."""
        logger.info("Starting Prometheus metrics server on port %d ...", self.port)
        start_http_server(self.port)

    # Example helpers to record metrics -----------------------------

    def inc_orders(self, status: str):
        self.orders_total.labels(status=status).inc()

    def inc_events(self, event_type: str):
        self.events_processed.labels(type=event_type).inc()

    def set_position_value(self, value: float):
        self.position_value.set(value)

    def set_uncertainty(self, decision: str, pct: float):
        self.cognitive_uncertainty.labels(decision=decision).set(pct)

    def observe_order_latency(self, seconds: float):
        self.order_latency.observe(seconds)


if __name__ == "__main__":
    # Manual smoke test
    mc = MetricsCollector({"port": 9108})
    mc.start()
    while True:
        mc.inc_orders("filled")
        mc.inc_events("market_tick")
        mc.set_position_value(1_000_000)
        mc.set_uncertainty("BUY", 12.3)
        mc.observe_order_latency(0.035)
        time.sleep(1)

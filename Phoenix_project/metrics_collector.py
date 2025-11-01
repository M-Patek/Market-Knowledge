from prometheus_client import start_http_server, Counter, Gauge, Histogram
import time
from typing import Dict, Any

from ..monitor.logging import get_logger

logger = get_logger(__name__)

class MetricsCollector:
    """
    A centralized service for managing and exposing system metrics
    via a Prometheus endpoint.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the MetricsCollector and defines the metrics.
        
        Args:
            config: Main system configuration.
        """
        self.config = config.get('metrics', {})
        self.port = self.config.get('prometheus_port', 8008)
        
        # --- Define Metrics ---
        
        # Counter (Monotonically increasing)
        self.events_processed_total = Counter(
            'phoenix_events_processed_total',
            'Total number of events processed',
            ['event_type', 'status'] # Labels
        )
        
        self.trades_executed_total = Counter(
            'phoenix_trades_executed_total',
            'Total number of (simulated) trades executed',
            ['symbol', 'direction']
        )

        # Gauge (Can go up or down)
        self.portfolio_value_usd = Gauge(
            'phoenix_portfolio_value_usd',
            'Current total portfolio value (equity)'
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

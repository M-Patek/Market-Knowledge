# events/stream_processor.py
from collections import deque
import math
import asyncio
import numpy as np
from typing import Protocol, List, Dict, Any, AsyncGenerator
import redis
import hashlib

# Assuming an observability module exists for alerts
# from observability import alert_manager

# --- [Sub-Task 2.2.1] News Client Abstraction ---
class BaseNewsClient(Protocol):
    """Abstract base class for a news fetching client."""
    async def fetch(self) -> List[Dict[str, Any]]:
        ...

class BenzingaClient(BaseNewsClient):
    """Placeholder implementation for a Benzinga Pro news client."""
    async def fetch(self) -> List[Dict[str, Any]]:
        return [{"source": "Benzinga", "headline": "BREAKING: Fed to raise rates by 25 bps"}]

class AlphaVantageNewsClient(BaseNewsClient):
    """Placeholder implementation for an Alpha Vantage news client."""
    async def fetch(self) -> List[Dict[str, Any]]:
        return [{"source": "AlphaVantage", "headline": "BREAKING: Fed to raise rates by 25 bps"}]

class RealTimeDQM:
    """
    Performs real-time Data Quality Management on a stream of market data
    using techniques like the exponentially weighted moving average (EWMA).
    """
    
    def __init__(self, alpha=0.1, threshold_stdevs=3.0):
        """
        Initializes the DQM validator.
        
        Args:
            alpha (float): The smoothing factor for the EWMA.
            threshold_stdevs (float): Number of standard deviations to set the anomaly threshold.
        """
        self.alpha = alpha
        self.threshold_stdevs = threshold_stdevs
        
        # Store EWMA and EWMSD (Exponentially Weighted Moving Standard Deviation) for each asset
        self.ewma = {}
        self.ewmsd = {}
        
        self.min_observations = 20 # Need at least 20 observations to start anomaly detection
        self.observation_count = defaultdict(int)

    def check_anomaly(self, event: Dict[str, Any]) -> bool:
        """
        Checks a single event (e.g., a trade) for anomalies.
        
        Returns:
            bool: True if the event is considered an anomaly, False otherwise.
        """
        # We only check events that have a price and asset
        if event.get('type') != 'trade' or 'price' not in event or 'asset' not in event:
            return False
            
        asset = event['asset']
        price = event['price']
        
        self.observation_count[asset] += 1
        
        # Initialize or update the EWMA
        if asset not in self.ewma:
            self.ewma[asset] = price
            self.ewmsd[asset] = 0 # Initial standard deviation is 0
            return False
        else:
            deviation = price - self.ewma[asset]
            self.ewma[asset] = self.ewma[asset] + self.alpha * deviation
            # Update EWMSD using an EWMA of the squared deviations
            self.ewmsd[asset] = math.sqrt(
                (1 - self.alpha) * (self.ewmsd[asset]**2 + self.alpha * deviation**2)
            )

        # Don't check for anomalies until we have enough data
        if self.observation_count[asset] < self.min_observations:
            return False
            
        # Check if the price is outside the dynamic threshold
        threshold = self.threshold_stdevs * self.ewmsd[asset]
        if abs(deviation) > threshold:
            # alert_manager.send_alert(
            #     "DataQualityAnomaly",
            #     f"Anomalous price for {asset}: {price}. "
            #     f"Expected range: {self.ewma[asset] - threshold} to {self.ewma[asset] + threshold}"
            # )
            return True
        
        return False


class StreamProcessor:
    """
    [Epic 2.2] Aggregates events from multiple sources, deduplicates them, and runs DQM checks.
    """
    def __init__(self, news_clients: List[BaseNewsClient], dqm_enabled: bool = True, config: Dict[str, Any] = None):
        self.news_clients = news_clients
        self.dqm_validator = RealTimeDQM() if dqm_enabled else None
        # --- [Sub-Task 2.2.2] Deduplication Mechanism ---
        self.redis_client = redis.Redis(host='localhost', port=6379, db=1, decode_responses=True) # Using db=1 to separate from queues
        self.dedupe_ttl = config.get('deduplication_ttl_seconds', 600) if config else 600

    async def _deduplicate_and_yield_events(self, all_events: List[Dict[str, Any]]) -> AsyncGenerator[Dict[str, Any], None]:
        """Deduplicates events and yields unique ones."""
        for event in all_events:
            # Normalize headline for more robust deduplication
            normalized_headline = event.get("headline", "").lower().strip()
            # Atomically set the key with a 10-minute TTL if it does not already exist (nx=True).
            # The command returns True if the key was set, False otherwise.
            if normalized_headline and self.redis_client.set(normalized_headline, 1, ex=self.dedupe_ttl, nx=True):
                yield event

    async def process_events(self) -> AsyncGenerator[Dict[str, Any], None]:
        """
        The main event processing generator. Fetches from all sources, deduplicates,
        validates, and yields a clean stream of events.
        """
        while True:
            fetch_tasks = [client.fetch() for client in self.news_clients]
            results = await asyncio.gather(*fetch_tasks)
            all_events = [event for sublist in results for event in sublist]

            async for unique_event in self._deduplicate_and_yield_events(all_events):
                if not self.dqm_validator or not self.dqm_validator.check_anomaly(unique_event):
                    yield unique_event
            
            await asyncio.sleep(10) # Poll every 10 seconds

import json
import time

# Configure logger for this module (Layer 12)
from observability import get_logger
logger = get_logger(__name__)

class DataManager:
    """
    Manages access to all data sources (e.g., historical, real-time, unstructured).
    """

    def __init__(self, catalog_path: str):
        self.catalog_path = catalog_path
        try:
            with open(catalog_path, 'r') as f:
                self.catalog = json.load(f)
        except FileNotFoundError:
            logger.error(f"Data catalog file not found: {catalog_path}")
            self.catalog = {"sample_events": []}

    def stream_data(self):
        """
        Simulates a real-time data stream by yielding events from the catalog.
        """
        # In a real app, this would connect to a live data stream (e.g., Kafka, WebSocket)
        # For simulation, we'll just yield the sample events from the catalog
        
        logger.info(f"DataManager: Streaming data from '{self.catalog_path}'...")
        
        for event in self.catalog.get('sample_events', []):
            logger.debug(f"DataManager: Yielding event for {event.get('ticker')}")
            yield event
            time.sleep(0.5) # Simulate real-time data flow

        logger.info("DataManager: End of data stream.")

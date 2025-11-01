import json
import time

# Configure logger for this module (Layer 12)
# FIXED: 'observability.py' 不存在。'get_logger' 位于 'monitor/logging.py'。
from monitor.logging import get_logger
logger = get_logger(__name__)

class DataManager:
# ... existing code ...
    def __init__(self, catalog_path: str):
        self.catalog_path = catalog_path
# ... existing code ...
            logger.error(f"Data catalog file not found: {catalog_path}")
            self.catalog = {"sample_events": []}

    def stream_data(self):
# ... existing code ...
        logger.info(f"DataManager: Streaming data from '{self.catalog_path}'...")
        
        for event in self.catalog.get('sample_events', []):
# ... existing code ...
            yield event
            time.sleep(0.5) # Simulate real-time data flow

        logger.info("DataManager: End of data stream.")

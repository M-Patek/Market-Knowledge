import pandas as pd
from elasticsearch import AsyncElasticsearch
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..monitor.logging import get_logger

logger = get_logger(__name__)

class TemporalDBClient:
    """
    Client for interacting with a temporal database (e.g., Elasticsearch).
    Used for storing and retrieving time-series events (news, filings, etc.).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the Elasticsearch client.
        
        Args:
            config (Dict[str, Any]): Configuration dict, expects 'temporal_db'
                                      with 'hosts' and 'index_name'.
        """
        db_config = config.get('temporal_db', {})
        self.es = AsyncElasticsearch(
            hosts=db_config.get('hosts', ["http://localhost:9200"])
        )
        self.index_name = db_config.get('index_name', 'market_events')
        logger.info(f"TemporalDBClient initialized for index: {self.index_name}")

    async def connect(self):
        """Checks the connection to Elasticsearch."""
        try:
            if not await self.es.ping():
                logger.error("Elasticsearch connection failed.")
                raise ConnectionError("Failed to connect to Elasticsearch")
            
            # Ensure index exists
            if not await self.es.indices.exists(index=self.index_name):
                logger.warning(f"Elasticsearch index '{self.index_name}' not found. Attempting to create.")
                # TODO: Add index mapping definition
                await self.es.indices.create(index=self.index_name)
                
            logger.info("Elasticsearch connection successful.")
        except Exception as e:
            logger.error(f"Error connecting to Elasticsearch: {e}", exc_info=True)
            raise

    async def close(self):
        """Closes the Elasticsearch connection."""
        await self.es.close()
        logger.info("Elasticsearch connection closed.")

    async def index_event(self, event_id: str, event_data: Dict[str, Any]) -> bool:
        """
        Indexes a single event document in Elasticsearch.
        
        Args:
            event_id (str): The unique ID for the document.
            event_data (Dict[str, Any]): The event data (must be JSON-serializable).
            
        Returns:
            bool: True on success, False on failure.
        """
        try:
            # Ensure timestamp is in correct format
            if 'timestamp' in event_data and isinstance(event_data['timestamp'], pd.Timestamp):
                event_data['timestamp'] = event_data['timestamp'].isoformat()
                
            response = await self.es.index(
                index=self.index_name,
                id=event_id,
                document=event_data
            )
            
            if response.get('result') not in ['created', 'updated']:
                logger.warning(f"Failed to index event {event_id}: {response}")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error indexing event {event_id} in Elasticsearch: {e}", exc_info=True)
            return False

    async def search_events(
        self, 
        symbols: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        query_string: Optional[str] = None,
        size: int = 25
    ) -> List[Dict[str, Any]]:
        """
        Searches the temporal database for events matching the criteria.
        
        Args:
            symbols: List of ticker symbols to match.
            start_time: The start of the time window.
            end_time: The end of the time window.
            query_string: A free-text query string.
            size: The maximum number of hits to return.
            
        Returns:
            List[Dict[str, Any]]: A list of event data dictionaries.
        """
        try:
            query = {"bool": {"filter": []}}
            
            # Time range filter
            time_range = {}
            if start_time:
                time_range["gte"] = start_time.isoformat()
            if end_time:
                time_range["lte"] = end_time.isoformat()
            if time_range:
                query["bool"]["filter"].append({"range": {"timestamp": time_range}})
                
            # Symbols filter (assuming 'symbols' is a keyword field)
            if symbols:
                query["bool"]["filter"].append({"terms": {"symbols": symbols}})
                
            # Full-text query (if provided)
            if query_string:
                query["bool"]["must"] = {
                    "query_string": {
                        "query": query_string,
                        "fields": ["headline", "summary", "tags"]
                    }
                }
            
            response = await self.es.search(
                index=self.index_name,
                query=query,
                size=size,
                sort=[{"timestamp": "desc"}] # Most recent first
            )
            
            hits = response.get('hits', {}).get('hits', [])
            return [hit.get('_source', {}) for hit in hits]
            
        except Exception as e:
            logger.error(f"Error searching Elasticsearch: {e}", exc_info=True)
            return []

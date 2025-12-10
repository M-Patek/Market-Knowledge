import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import pandas as pd
from elasticsearch import AsyncElasticsearch, helpers

logger = logging.getLogger(__name__)

class TemporalDBClient:
    """
    Client for interacting with the Temporal Database (Elasticsearch/OpenSearch).
    Stores and retrieves time-series events (news, market data, signals).
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        self.host = self.config.get("host", "phoenix_elasticsearch")
        self.port = self.config.get("port", 9200)
        self.username = self.config.get("user_env") or os.environ.get("ELASTIC_USER", "elastic")
        self.password = self.config.get("pass_env") or os.environ.get("ELASTIC_PASSWORD", "changeme")
        self.index_name = self.config.get("index_name", "phoenix-temporal-events")
        
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        try:
            # Construct URI or use explicit host/port
            # For simplicity using basic auth with host/port
            self.client = AsyncElasticsearch(
                hosts=[f"http://{self.host}:{self.port}"],
                basic_auth=(self.username, self.password),
                verify_certs=False # In internal docker network usually fine
            )
            logger.info(f"TemporalDBClient initialized for {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to initialize TemporalDBClient: {e}")

    async def query_market_data(self, symbol: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        Retrieves market data (OHLCV) for a symbol within a time range.
        Returns a Pandas DataFrame.
        """
        if not self.client:
            return pd.DataFrame()

        query = {
            "bool": {
                "must": [
                    {"term": {"symbol": symbol}},
                    {"term": {"event_type": "market_data"}},
                    {"range": {"timestamp": {"gte": start_time.isoformat(), "lte": end_time.isoformat()}}}
                ]
            }
        }
        
        sort_criteria = [{"timestamp": {"order": "asc"}}]
        batch_size = 1000 # Pagination size

        try:
            all_source_data = []
            last_sort_values = None

            while True:
                # Prepare search args
                search_args = {
                    "index": self.index_name,
                    "query": query,
                    "sort": sort_criteria,
                    "size": batch_size
                }
                if last_sort_values:
                    search_args["search_after"] = last_sort_values

                # Execute search
                resp = await self.client.search(**search_args)
                
                hits = resp.get('hits', {}).get('hits', [])
                if not hits:
                    break

                # Collect data
                for h in hits:
                    all_source_data.append(h['_source'])
                
                # Update for next page
                last_hit = hits[-1]
                if 'sort' in last_hit:
                    last_sort_values = last_hit['sort']
                else:
                    break # Should not happen with explicit sort

                if len(hits) < batch_size:
                    break

            if not all_source_data:
                return pd.DataFrame()

            # Parse results
            df = pd.DataFrame(all_source_data)
            
            # Ensure timestamp is datetime and set as index
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            return df

        except Exception as e:
            logger.error(f"TemporalDB query_market_data failed: {e}")
            return pd.DataFrame()

    async def query_events(self, event_type: str, start_time: datetime, end_time: datetime, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Generic event query (e.g., news, signals) within a time window.
        """
        if not self.client:
            return []

        query = {
            "bool": {
                "must": [
                    {"term": {"event_type": event_type}},
                    {"range": {"timestamp": {"gte": start_time.isoformat(), "lte": end_time.isoformat()}}}
                ]
            }
        }
        
        sort_criteria = [{"timestamp": {"order": "desc"}}]
        batch_size = 1000 
        # Note: If limit is smaller than batch_size, we can just use limit.
        # But if user asks for more than 10k (default max_result_window), pagination is needed.
        # Here we assume limit can be large, so we paginate until limit is reached.
        
        effective_batch_size = min(limit, batch_size)

        try:
            all_source_data = []
            last_sort_values = None
            
            while len(all_source_data) < limit:
                current_size = min(batch_size, limit - len(all_source_data))
                
                search_args = {
                    "index": self.index_name,
                    "query": query,
                    "sort": sort_criteria,
                    "size": current_size
                }
                if last_sort_values:
                    search_args["search_after"] = last_sort_values
                
                resp = await self.client.search(**search_args)
                
                hits = resp.get('hits', {}).get('hits', [])
                if not hits:
                    break
                    
                for h in hits:
                    all_source_data.append(h['_source'])
                    
                last_hit = hits[-1]
                if 'sort' in last_hit:
                    last_sort_values = last_hit['sort']
                else:
                    break
                
                if len(hits) < current_size:
                    break

            return all_source_data[:limit]

        except Exception as e:
            logger.error(f"TemporalDB query_events failed: {e}")
            return []
            
    async def ingest_batch(self, events: List[Dict[str, Any]]) -> bool:
        """
        Bulk ingest a list of events.
        """
        if not self.client or not events:
            return False
            
        actions = []
        for event in events:
            action = {
                "_index": self.index_name,
                "_source": event
            }
            # Optional: Deduplication ID generation if needed
            # if 'id' in event: action['_id'] = event['id']
            actions.append(action)
            
        try:
            await helpers.async_bulk(self.client, actions)
            return True
        except Exception as e:
            logger.error(f"TemporalDB ingest_batch failed: {e}")
            return False

    async def close(self):
        """[Fix] Resource Cleanup: Close database connection pool."""
        if hasattr(self, 'client') and self.client:
            await self.client.close()

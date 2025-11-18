import pandas as pd
from elasticsearch import AsyncElasticsearch
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# 修复：将相对导入 'from ..monitor.logging...' 更改为绝对导入
from Phoenix_project.monitor.logging import get_logger
from ..utils.retry import retry_with_exponential_backoff

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
        # [蓝图 2 修复]：配置现在是 config['temporal_db']
        db_config = config # 假设 config 已经是 config['temporal_db']
        self.es = AsyncElasticsearch(
            hosts=db_config.get('hosts', ["http://localhost:9200"])
        )
        self.index_name = db_config.get('index_name', 'market_events')
        logger.info(f"TemporalDBClient initialized for index: {self.index_name}")

    @retry_with_exponential_backoff()
    async def connect(self):
        """Checks the connection to Elasticsearch."""
        try:
            if not await self.es.ping():
                logger.error("Elasticsearch connection failed.")
                raise ConnectionError("Failed to connect to Elasticsearch")
            
            # Ensure index exists
            if not await self.es.indices.exists(index=self.index_name):
                logger.warning(f"Elasticsearch index '{self.index_name}' not found. Attempting to create.")
                
                # --- [任务 1 已添加] ---
                # 定义索引映射 (Index Mapping)
                index_mapping = {
                    "mappings": {
                        "properties": {
                            "timestamp": {"type": "date"},
                            "symbols": {"type": "keyword"},
                            "headline": {"type": "text", "analyzer": "standard"},
                            "summary": {"type": "text", "analyzer": "standard"},
                            "content": {"type": "text", "analyzer": "standard"},
                            "tags": {"type": "keyword"},
                            "source": {"type": "keyword"}
                        }
                    }
                }
                logger.info(f"Creating index '{self.index_name}' with mapping.")
                await self.es.indices.create(index=self.index_name, body=index_mapping)
                # --- [任务 1 结束] ---
                
            logger.info("Elasticsearch connection successful.")
        except Exception as e:
            logger.error(f"Error connecting to Elasticsearch: {e}", exc_info=True)
            raise

    async def close(self):
        """Closes the Elasticsearch connection."""
        await self.es.close()
        logger.info("Elasticsearch connection closed.")

    @retry_with_exponential_backoff()
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
            if 'timestamp' in event_data and isinstance(event_data['timestamp'], (pd.Timestamp, datetime)):
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

    @retry_with_exponential_backoff()
    async def search_events(
        self, 
        symbols: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        query_string: Optional[str] = None,
        size: int = 25
    ) -> List[Tuple[Dict[str, Any], float]]: # <--- [蓝图 2] 更改返回类型
        """
        Searches the temporal database for events matching the criteria.
        
        Args:
            symbols: List of ticker symbols to match.
            start_time: The start of the time window.
            end_time: The end of the time window.
            query_string: A free-text query string.
            size: The maximum number of hits to return.
            
        Returns:
            List[Tuple[Dict[str, Any], float]]: A list of (event_data, score) tuples.
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
                        # 假设这些字段在 ES 映射中存在
                        "fields": ["headline", "summary", "tags", "content"] # <--- [蓝图 2] 添加 'content'
                    }
                }
            
            # 如果没有 query_string，我们仍然希望有结果，但按时间排序
            sort_order = [{"_score": "desc"}, {"timestamp": "desc"}]
            if not query_string:
                sort_order = [{"timestamp": "desc"}] # 如果没有查询，则按时间排序

            response = await self.es.search(
                index=self.index_name,
                query=query,
                size=size,
                sort=sort_order # <--- [蓝图 2] 按分数排序
            )
            
            hits = response.get('hits', {}).get('hits', [])
            
            # [蓝图 2] 提取 _source 和 _score
            results_with_scores = []
            for hit in hits:
                source_data = hit.get('_source', {})
                # 如果没有提供 query_string，ES 可能不会返回 _score，或者返回 0.0
                score = float(hit.get('_score') or 1.0) # 如果没有分数则默认为 1.0
                results_with_scores.append((source_data, score))
                
            return results_with_scores
            
        except Exception as e:
            logger.error(f"Error searching Elasticsearch: {e}", exc_info=True)
            return []

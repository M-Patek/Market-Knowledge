from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
# import aiofiles # [主人喵的清洁计划 1.3] 移除
import json
# import os # [主人喵的清洁计划 1.3] 移除

# 修正：将 'core.schemas...' 转换为 'Phoenix_project.core.schemas...'
from Phoenix_project.core.schemas.fusion_result import FusionResult
# 修正：将 'monitor.logging...' 转换为 'Phoenix_project.monitor.logging...'
from Phoenix_project.monitor.logging import ESLogger, get_logger

# [主人喵的清洁计划 1.3] 导入 ES 客户端
try:
    from elasticsearch import Elasticsearch, AsyncElasticsearch
    from elasticsearch.helpers import async_bulk
except ImportError:
    # 此时 logger 可能尚未初始化
    print("ERROR: Elasticsearch client not found. `pip install elasticsearch`")
    Elasticsearch = None
    AsyncElasticsearch = None


class CoTDatabase:
    """
    [主人喵的清洁计划 1.3] 已重构
    使用 Elasticsearch 存储和检索思维链 (CoT) 推理轨迹。
    (注意：这替代了基于文件系统的实现，以匹配 system.yaml 和 Janitor)
    """

    def __init__(self, config: Dict[str, Any], logger: ESLogger):
        """
        初始化 CoTDatabase (Elasticsearch)。
        """
        self.config = config
        self.logger = logger or get_logger(__name__)

        if Elasticsearch is None:
            self.logger.log_critical("Elasticsearch client is not installed. CoTDatabase will not function.")
            self.es_client = None
            self.async_es_client = None
            return

        try:
            # 从 system.yaml ai.cot_database 获取配置
            host = self.config.get("host", "localhost")
            port = self.config.get("port", 9200)
            self.index_name = self.config.get("index", "phoenix-cot-traces") # 假设一个索引
            es_url = f"http://{host}:{port}"
            
            self_options = {"max_retries": 3, "retry_on_timeout": True}
            # 同步客户端 (用于 Janitor)
            self.es_client = Elasticsearch(es_url, **self_options)
            # 异步客户端 (用于 App)
            self.async_es_client = AsyncElasticsearch(es_url, **self_options)
            
            if not self.es_client.indices.exists(index=self.index_name):
                self.logger.log_info(f"Creating Elasticsearch CoT index: {self.index_name}")
                self.es_client.indices.create(index=self.index_name, body={
                    "mappings": {
                        "properties": {
                            "timestamp": {"type": "date"},
                            "decision_id": {"type": "keyword"},
                            "decision": {"type": "keyword"},
                            "reasoning": {"type": "text"},
                            "confidence": {"type": "float"},
                            "metadata": {"type": "object", "enabled": False} # 减少索引字段
                        }
                    }
                })
            self.logger.log_info(f"CoTDatabase initialized. Storage path: ES index {self.index_name}")

        except Exception as e:
            self.logger.log_error(f"Failed to initialize CoTDatabase ES client: {e}", exc_info=True)
            self.es_client = None
            self.async_es_client = None

    async def store_trace(
        self, event_id: str, trace_data: Dict[str, Any]
    ) -> bool:
        """
        [主人喵的清洁计划 1.3] 存储轨迹到 ES
        """
        if not self.async_es_client:
            self.logger.log_error("Async ES client not initialized. Cannot store trace.")
            return False
        if not event_id:
            self.logger.log_warning("store_trace called with empty event_id. Skipping.")
            return False

        try:
            # event_id 用作 ES 文档 ID
            await self.async_es_client.index(
                index=self.index_name,
                id=event_id,
                document=trace_data
            )
            self.logger.log_info(f"Successfully stored trace for event {event_id}.")
            return True
        except Exception as e:
            self.logger.log_error(
                f"Failed to store trace for event {event_id}: {e}", exc_info=True
            )
            return False

    async def retrieve_trace(self, event_id: str) -> Optional[Dict[str, Any]]:
        """
        [主人喵的清洁计划 1.3] 从 ES 检索轨迹
        """
        if not self.async_es_client:
            self.logger.log_error("Async ES client not initialized. Cannot retrieve trace.")
            return None
            
        try:
            response = await self.async_es_client.get(
                index=self.index_name,
                id=event_id
            )
            return response.get("_source")
        except Exception as e: # 尤其是 NotFoundError
            self.logger.log_warning(f"Failed to retrieve trace {event_id} (may not exist): {e}")
            return None

    async def search_traces(self, keywords: List[str], limit: int = 5) -> List[Dict[str, Any]]:
        """
        [主人喵的清洁计划 1.3] 在 ES 中搜索轨迹
        """
        if not self.async_es_client:
            self.logger.log_error("Async ES client not initialized. Cannot search traces.")
            return []
            
        self.logger.log_debug(f"Searching traces for keywords: {keywords} (limit {limit})")
        if not keywords:
            return []
        
        try:
            query = {
                "multi_match": {
                    "query": " ".join(keywords),
                    "fields": ["reasoning", "decision", "metadata.*"] # 搜索关键字段
                }
            }
            response = await self.async_es_client.search(
                index=self.index_name,
                query=query,
                size=limit
            )
            return [hit["_source"] for hit in response["hits"]["hits"]]
        except Exception as e:
            self.logger.log_error(f"Failed during search_traces: {e}", exc_info=True)
            return []

    async def query_by_time(
        self, start_time: datetime, end_time: datetime, limit: int
    ) -> List[Dict[str, Any]]:
        """
        [主人喵的清洁计划 1.3] 按时间查询轨迹
        """
        if not self.async_es_client:
            self.logger.log_error("Async ES client not initialized. Cannot query by time.")
            return []
            
        try:
            query = {
                "range": {
                    "timestamp": {
                        "gte": start_time.isoformat(),
                        "lte": end_time.isoformat()
                    }
                }
            }
            response = await self.async_es_client.search(
                index=self.index_name,
                query=query,
                size=limit,
                sort=[{"timestamp": "desc"}] # 获取最新的
            )
            return [hit["_source"] for hit in response["hits"]["hits"]]
        except Exception as e:
            self.logger.log_error(f"Failed during query_by_time: {e}", exc_info=True)
            return []

    async def get_all_keys(self) -> List[str]:
        """
        [主人喵的清洁计划 1.3] 获取所有键 (ID)
        警告：在大型索引上可能非常慢。
        """
        if not self.async_es_client:
            self.logger.log_error("Async ES client not initialized. Cannot get all keys.")
            return []
            
        self.logger.log_warning("get_all_keys is inefficient on large ES indices.")
        try:
            # 使用 _search 只返回 _id
            response = await self.async_es_client.search(
                index=self.index_name,
                _source=False,
                size=10000 # 限制数量
            )
            return [hit["_id"] for hit in response["hits"]["hits"]]
        except Exception as e:
            self.logger.log_error(f"Failed during get_all_keys: {e}", exc_info=True)
            return []

    # [主人喵的清洁计划 1.3] Janitor 需要的方法
    def delete_by_query(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """
        (同步) Deletes CoT logs by a specific query.
        Used by the System Janitor (as expected by scripts/run_system_janitor.py).
        """
        if not self.es_client:
            self.logger.log_error("Sync ES client not initialized. Cannot run delete_by_query.")
            return {"error": "ES client not initialized"}
        
        try:
            self.logger.info(f"Janitor: Running delete_by_query on {self.index_name}...")
            response = self.es_client.delete_by_query(
                index=self.index_name,
                body=body,
                wait_for_completion=True,
                refresh=True
            )
            self.logger.info(f"Janitor: Delete complete: {response}")
            return response
        except Exception as e:
            self.logger.log_error(f"Janitor: Failed to delete old CoT logs: {e}", exc_info=True)
            return {"error": str(e)}

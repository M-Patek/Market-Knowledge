import json
# import aiofiles # [主人喵的清洁计划 1.2] 移除
from datetime import datetime
from typing import Dict, Any, Optional
# from Phoenix_project.config.loader import ConfigLoader # [主人喵的清洁计划 1.2] 移除
from Phoenix_project.monitor.logging import get_logger as get_system_logger

# [主人喵的清洁计划 1.2] 导入 ES 客户端
try:
    from elasticsearch import Elasticsearch, AsyncElasticsearch
    from elasticsearch.helpers import async_bulk
except ImportError:
    print("ERROR: Elasticsearch client not found. `pip install elasticsearch`")
    Elasticsearch = None
    AsyncElasticsearch = None

system_logger = get_system_logger(__name__)

class AuditLogger:
    """
    [主人喵的清洁计划 1.2] 已重构
    Handles logging of critical decisions and system states to Elasticsearch
    for auditing and traceability.
    """

    def __init__(self, config: Dict[str, Any]): # [主人喵的清洁计划 1.2] 更改 __init__ 以匹配 Janitor 的调用
        self.config = config
        
        if Elasticsearch is None:
            system_logger.critical("Elasticsearch client is not installed. AuditLogger will not function.")
            self.es_client = None
            self.async_es_client = None
            return
            
        try:
            # [主人喵的清洁计划 1.2] 从 Janitor 脚本的假设中获取配置
            host = self.config.get("host", "localhost")
            port = self.config.get("port", 9200)
            self.index_name = self.config.get("index", "phoenix-audit-logs") # Janitor 中指定的索引
            es_url = f"http://{host}:{port}"
            
            # 同步客户端 (用于 Janitor)
            self_options = {"max_retries": 3, "retry_on_timeout": True}
            self.es_client = Elasticsearch(es_url, **self_options)
            # 异步客户端 (用于 App)
            self.async_es_client = AsyncElasticsearch(es_url, **self_options)
            
            # 确保索引存在
            if not self.es_client.indices.exists(index=self.index_name):
                system_logger.info(f"Creating Elasticsearch audit index: {self.index_name}")
                self.es_client.indices.create(index=self.index_name, body={
                    "mappings": {
                        "properties": {
                            "timestamp": {"type": "date"},
                            "decision_id": {"type": "keyword"},
                            "event_type": {"type": "keyword"},
                            "component": {"type": "keyword"}
                        }
                    }
                })
            system_logger.info(f"AuditLogger initialized. Logging to ES index: {self.index_name}")
            
        except Exception as e:
            system_logger.error(f"Failed to initialize Elasticsearch for AuditLogger: {e}", exc_info=True)
            self.es_client = None
            self.async_es_client = None

    async def log_event(
        self,
        event_type: str,
        details: Dict[str, Any],
        pipeline_state: Optional[Any] = None, # Avoid circular import, type as Any
        decision_id: Optional[str] = None
    ):
        """
        [主人喵的清洁计划 1.2] 已重构
        Asynchronously logs a structured audit event to Elasticsearch.
        """
        
        if not self.async_es_client:
            system_logger.error("Async ES client not initialized. Cannot log audit event.")
            return
            
        timestamp = datetime.utcnow() # ES a/sync client handles isoformat
        
        log_entry = {
            "timestamp": timestamp,
            "decision_id": decision_id or "N/A",
            "event_type": event_type,
            "details": details,
        }
        
        # Optionally snapshot key parts of the state
        if pipeline_state:
            try:
                # Be selective to avoid logging huge amounts of data
                log_entry["state_snapshot"] = {
                    "current_time": pipeline_state.get_value("current_time"),
                    "last_decision": pipeline_state.get_value("last_decision"),
                    # Add other key fields, but avoid full context data
                }
            except Exception as e:
                system_logger.warning(f"Failed to create state snapshot for audit log: {e}")
                log_entry["state_snapshot"] = {"error": "Failed to serialize state"}

        try:
            # [主人喵的清洁计划 1.2] 写入 ES
            await self.async_es_client.index(
                index=self.index_name,
                document=log_entry
            )
                
        except Exception as e:
            system_logger.error(f"FATAL: Failed to write to audit ES index: {e}", exc_info=True)
            # In a production system, this might trigger a circuit breaker

    async def log_decision(
        self,
        decision_id: str,
        fusion_result: Any, # FusionResult
        pipeline_state: Any  # PipelineState
    ):
        """Helper method to specifically log a final decision."""
        details = {
            "message": "Final decision synthesized.",
            "decision": fusion_result.final_decision if hasattr(fusion_result, 'final_decision') else 'N/A',
            "confidence": fusion_result.confidence if hasattr(fusion_result, 'confidence') else 0.0,
            "reasoning": fusion_result.reasoning if hasattr(fusion_result, 'reasoning') else 'N/A',
            "contributing_agents": [
                agent.model_dump() for agent in fusion_result.contributing_agents
            ] if hasattr(fusion_result, 'contributing_agents') and fusion_result.contributing_agents else []
        }
        await self.log_event(
            event_type="DECISION_SYNTHESIS",
            details=details,
            pipeline_state=pipeline_state,
            decision_id=decision_id
        )

    async def log_error(
        self,
        error_message: str,
        component: str,
        pipeline_state: Optional[Any] = None,
        decision_id: Optional[str] = None
    ):
        """Helper method to log a critical error."""
        details = {
            "message": "A critical error occurred.",
            "component": component,
            "error": error_message
        }
        await self.log_event(
            event_type="CRITICAL_ERROR",
            details=details,
            pipeline_state=pipeline_state,
            decision_id=decision_id
        )

    # [主人喵的清洁计划 1.2] 新增 Janitor 方法
    def delete_older_than(self, cutoff_date: datetime) -> Dict[str, Any]:
        """
        [主人喵的清洁计划 1.2]
        (同步) Deletes audit logs older than the specified cutoff date.
        Used by the System Janitor.
        """
        if not self.es_client:
            system_logger.error("Sync ES client not initialized. Cannot run delete_older_than.")
            return {"error": "ES client not initialized"}
            
        body = {
            "query": {
                "range": {
                    "timestamp": { # 假设 log_event 使用 'timestamp'
                        "lt": cutoff_date.isoformat()
                    }
                }
            }
        }
        
        try:
            system_logger.info(f"Janitor: Deleting documents from {self.index_name} older than {cutoff_date.isoformat()}...")
            response = self.es_client.delete_by_query(
                index=self.index_name,
                body=body,
                wait_for_completion=True,
                refresh=True
            )
            system_logger.info(f"Janitor: Delete complete: {response}")
            return response
        except Exception as e:
            system_logger.error(f"Janitor: Failed to delete old audit logs: {e}", exc_info=True)
            return {"error": str(e)}

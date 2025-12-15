from typing import Dict, Any, Optional
import uuid
from copy import deepcopy
from Phoenix_project.audit.logger import AuditLogger
from Phoenix_project.memory.cot_database import CoTDatabase
from Phoenix_project.core.schemas.fusion_result import FusionResult
from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

class AuditManager:
    """
    Coordinates all auditing activities.
    - Generates unique IDs for decisions.
    - Logs events to the persistent AuditLogger (file/db).
    - Stores Chain-of-Thought (CoT) reasoning in the CoTDatabase.
    [Task P2-002] Sensitive data sanitization implemented.
    """

    def __init__(self, audit_logger: AuditLogger, cot_database: CoTDatabase):
        self.audit_logger = audit_logger
        self.cot_database = cot_database
        logger.info("AuditManager initialized.")

    def generate_decision_id(self) -> str:
        """Generates a unique ID for a single reasoning cycle."""
        return f"dec_{uuid.uuid4()}"

    def _sanitize_data(self, data: Any) -> Any:
        """
        [Task P2-002] 递归脱敏敏感信息。
        查找键名包含 key, secret, password, token 的字段，并替换其值。
        """
        if not isinstance(data, (dict, list)):
            return data

        # Deep copy to ensure the original payload is not modified in memory
        sanitized_data = deepcopy(data) 

        if isinstance(sanitized_data, dict):
            for key, value in sanitized_data.items():
                lower_key = key.lower()
                
                # Check for sensitive keywords
                is_sensitive = any(
                    kw in lower_key for kw in ['key', 'secret', 'password', 'token']
                )

                if is_sensitive and isinstance(value, str):
                    # Replace value with mask
                    sanitized_data[key] = "******"
                elif isinstance(value, (dict, list)):
                    # Recurse into nested structures
                    sanitized_data[key] = self._sanitize_data(value)
        
        elif isinstance(sanitized_data, list):
            # Recurse into list elements
            for i in range(len(sanitized_data)):
                sanitized_data[i] = self._sanitize_data(sanitized_data[i])

        return sanitized_data

    async def audit_decision_cycle(
        self,
        pipeline_state: PipelineState,
        fusion_result: FusionResult,
        decision_id: str
    ):
        """
        Logs the complete record of a decision cycle to all audit stores.
        
        Args:
            pipeline_state (PipelineState): The state *after* the decision was made.
            fusion_result (FusionResult): The final decision object.
            decision_id (str): The unique ID for this cycle.
        """
        logger.info(f"Auditing decision cycle: {decision_id}")
        
        # 1. Log the full decision to the persistent audit trail (JSONL)
        try:
            # Note: PipelineState and FusionResult are structured Pydantic models; 
            # they should not contain raw secrets unless they are logged as a dictionary 
            # within a key/value pair. Sanitization of generic 'details' in audit_event is primary.
            await self.audit_logger.log_decision(
                decision_id=decision_id,
                fusion_result=fusion_result,
                pipeline_state=pipeline_state
            )
        except Exception as e:
            logger.error(f"Failed to log decision to audit trail: {e}", exc_info=True)
            # This is a critical failure

        # 2. Store the reasoning trace (CoT) in the Vector DB for retrieval
        try:
            # (FIX) 构建 trace_data 字典以匹配 store_trace
            trace_data = {
                "decision_id": decision_id,
                # (FIX) 调用新添加的 (已修复的) 方法
                "context": pipeline_state.get_full_context_formatted(),
                # (FIX) 直接访问属性
                "timestamp": pipeline_state.current_time.isoformat(),
                # (FIX) 访问 FusionResult 字段
                "decision": fusion_result.decision,
                "reasoning": fusion_result.reasoning,
                "confidence": float(fusion_result.confidence), # 确保是 float
                "metadata": {
                    # (FIX) 使用 get_value 作为安全访问器
                    "cycle_time_ms": pipeline_state.get_value("last_cycle_time_ms"),
                    "uncertainty": float(fusion_result.uncertainty),
                    "supporting_evidence_ids": fusion_result.supporting_evidence_ids,
                    "conflicting_evidence_ids": fusion_result.conflicting_evidence_ids,
                    "contributing_agents": [item.agent_id for item in fusion_result.agent_decisions] if fusion_result.agent_decisions else []
                }
            }
            
            # (FIX) 调用 CoTDatabase 上的正确方法
            await self.cot_database.store_trace(
                event_id=decision_id,
                trace_data=trace_data
            )
        except Exception as e:
            logger.error(f"Failed to store CoT trace in database: {e}", exc_info=True)
            # This is also critical, as it breaks the self-correction loop

    async def audit_event(
        self,
        event_type: str,
        details: Dict[str, Any],
        pipeline_state: Optional[PipelineState] = None,
        decision_id: Optional[str] = None
    ):
        """Logs a generic event (e.g., error, config change) to the audit trail."""
        if not decision_id and pipeline_state:
            # (FIX) 使用 get_value 作为安全访问器
            decision_id = pipeline_state.get_value("current_decision_id")
        
        # [TASK-P2-002 Fix] Sanitize the details payload before logging
        sanitized_details = self._sanitize_data(details)
            
        await self.audit_logger.log_event(
            event_type=event_type,
            details=sanitized_details,
            pipeline_state=pipeline_state,
            decision_id=decision_id
        )

    async def audit_error(
        self,
        error: Exception,
        component: str,
        pipeline_state: Optional[PipelineState] = None,
        decision_id: Optional[str] = None
    ):
        """Logs a critical error to the audit trail."""
        if not decision_id and pipeline_state:
            # (FIX) 使用 get_value 作为安全访问器
            decision_id = pipeline_state.get_value("current_decision_id")

        await self.audit_logger.log_error(
            error_message=str(error),
            component=component,
            pipeline_state=pipeline_state,
            decision_id=decision_id
        )

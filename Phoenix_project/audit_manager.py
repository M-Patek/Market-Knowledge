from typing import Dict, Any, Optional
import uuid
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
    - Stores Chain-of-Thought (CoT) reasoning in the CoTDatabase (vector db).
    """

    def __init__(self, audit_logger: AuditLogger, cot_database: CoTDatabase):
        self.audit_logger = audit_logger
        self.cot_database = cot_database
        logger.info("AuditManager initialized.")

    def generate_decision_id(self) -> str:
        """Generates a unique ID for a single reasoning cycle."""
        return f"dec_{uuid.uuid4()}"

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
            # We store the final *synthesized* reasoning, as it's the
            # most complete explanation for the action taken.
            await self.cot_database.add_reasoning_trace(
                decision_id=decision_id,
                timestamp=pipeline_state.get_value("current_time"),
                context=pipeline_state.get_full_context_formatted(), # Assumes this method exists
                decision=fusion_result.final_decision,
                reasoning=fusion_result.reasoning,
                confidence=fusion_result.confidence,
                metadata={
                    "cycle_time_ms": pipeline_state.get_value("last_cycle_time_ms"),
                    "arbitrator_suggestion": pipeline_state.get_value("last_arbitration", {}).get("suggested_decision")
                }
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
            decision_id = pipeline_state.get_value("current_decision_id")
            
        await self.audit_logger.log_event(
            event_type=event_type,
            details=details,
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
            decision_id = pipeline_state.get_value("current_decision_id")

        await self.audit_logger.log_error(
            error_message=str(error),
            component=component,
            pipeline_state=pipeline_state,
            decision_id=decision_id
        )

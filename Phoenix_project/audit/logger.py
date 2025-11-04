import json
import aiofiles
from datetime import datetime
from typing import Dict, Any, Optional
from Phoenix_project.config.loader import ConfigLoader
from Phoenix_project.monitor.logging import get_logger as get_system_logger

system_logger = get_system_logger(__name__)

class AuditLogger:
    """
    Handles logging of critical decisions and system states for auditing
    and traceability. Logs are typically written to a persistent,
    append-only file or database.
    """

    def __init__(self, config_loader: ConfigLoader):
        self.config = config_loader.get_config("system")
        self.log_path = self.config.get("audit_log_path", "logs/audit_trail.jsonl")
        
        # Ensure log directory exists
        try:
            import os
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        except Exception as e:
            system_logger.error(f"Failed to create audit log directory: {e}", exc_info=True)
            
        system_logger.info(f"AuditLogger initialized. Logging to: {self.log_path}")

    async def log_event(
        self,
        event_type: str,
        details: Dict[str, Any],
        pipeline_state: Optional[Any] = None, # Avoid circular import, type as Any
        decision_id: Optional[str] = None
    ):
        """
        Asynchronously logs a structured audit event.
        
        Args:
            event_type (str): The type of event (e.g., "DECISION", "ERROR", "STATE_CHANGE").
            details (Dict[str, Any]): A JSON-serializable dictionary of event details.
            pipeline_state (Optional[PipelineState]): The state at the time of the event.
            decision_id (Optional[str]): A unique ID to correlate events.
        """
        
        timestamp = datetime.utcnow().isoformat()
        
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
            async with aiofiles.open(self.log_path, mode='a') as f:
                # Convert to JSON string and add a newline
                await f.write(json.dumps(log_entry) + "\n")
                
        except Exception as e:
            system_logger.error(f"FATAL: Failed to write to audit log file: {e}", exc_info=True)
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
            "decision": fusion_result.final_decision,
            "confidence": fusion_result.confidence,
            "reasoning": fusion_result.reasoning,
            "contributing_agents": [
                agent.model_dump() for agent in fusion_result.contributing_agents
            ]
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

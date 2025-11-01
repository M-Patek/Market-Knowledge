from typing import Dict, Any
import pandas as pd
import json

from ..monitor.logging import get_logger

# Use the central logging setup
base_logger = get_logger(__name__)

class AuditLogger:
    """
    A specialized logger for writing structured audit trail data.
    This is distinct from application/debug logging.
    
    It outputs structured logs (e.g., JSON) to a dedicated file or stream,
    which can be consumed by analysis or compliance tools.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the AuditLogger.
        
        Args:
            config (Dict[str, Any]): Expects an 'audit_logger' config block.
        """
        audit_config = config.get('audit_logger', {})
        self.log_path = audit_config.get('log_path', 'logs/phoenix_audit.jsonl')
        
        # TODO: Set up a dedicated file handler for this logger
        # For simplicity in this example, we'll just log to a file manually,
        # but a real implementation should use the 'logging' framework
        # with a custom formatter and file handler.
        
        base_logger.info(f"AuditLogger initialized. Writing audit trail to: {self.log_path}")

    def _write_audit_log(self, log_type: str, data: Dict[str, Any]):
        """
        Internal method to write a structured JSON line to the audit log.
        
        Args:
            log_type (str): The type of event (e.g., "TRADE", "PIPELINE_IO").
            data (Dict[str, Any]): The payload to log.
        """
        try:
            # Ensure all data is JSON serializable
            log_entry = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "log_type": log_type,
                "data": self._sanitize_for_json(data)
            }
            
            with open(self.log_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            base_logger.error(f"Failed to write to audit log! Type: {log_type}. Error: {e}")

    def _sanitize_for_json(self, data: Any) -> Any:
        """
        Recursively sanitizes data to ensure it's JSON serializable.
        Converts pandas Timestamps, numpy objects, etc.
        """
        if isinstance(data, dict):
            return {k: self._sanitize_for_json(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._sanitize_for_json(v) for v in data]
        if isinstance(data, (pd.Timestamp, pd.Period)):
            return data.isoformat()
        if hasattr(data, 'to_dict'): # e.g., Pydantic models (if .model_dump() exists)
            try:
                return data.model_dump()
            except:
                pass # Fallback
        if hasattr(data, 'isoformat'): # e.g., datetime
             return data.isoformat()
        # Add more types as needed (numpy, etc.)

        # If simple type, return as is
        return data


    # --- Public Logging Methods ---

    def log_event_in(self, event: Dict[str, Any]):
        """Logs a raw event received by the system."""
        self._write_audit_log("EVENT_IN", {"event": event})

    def log_pipeline_io(self, event_id: str, pipeline_io: Dict[str, Any]):
        """Logs the full I/O bundle (RAG, Agent I/O) for a cognitive pipeline run."""
        self._write_audit_log("PIPELINE_IO", {"event_id": event_id, "io_bundle": pipeline_io})

    def log_fusion_result(self, fusion_result: Dict[str, Any]):
        """Logs the final decision and uncertainty from the cognitive engine."""
        self._write_audit_log("FUSION_RESULT", {"result": fusion_result})

    def log_signal(self, signal: Dict[str, Any]):
        """Logs the generated strategy signal (target weights)."""
        self._write_audit_log("STRATEGY_SIGNAL", {"signal": signal})

    def log_trade(self, fill_record: Dict[str, Any]):
        """Logs an executed (simulated) trade."""
        self._write_audit_log("TRADE", {"fill": fill_record})

    def log_costs(self, costs: Dict[str, Any], timestamp: pd.Timestamp):
        """Logs simulated execution costs."""
        log_data = costs.copy()
        log_data["log_timestamp"] = timestamp
        self._write_audit_log("EXECUTION_COSTS", log_data)

    def log_portfolio_snapshot(self, portfolio: Dict[str, float], pnl_record: Dict[str, Any]):
        """Logs a snapshot of the portfolio and PnL."""
        self._write_audit_log("PORTFOLIO_SNAPSHOT", {
            "positions": portfolio,
            "pnl": pnl_record
        })

    def log_system_error(self, error: Exception, context: str = "general"):
        """Logs a critical system error."""
        self._write_audit_log("SYSTEM_ERROR", {
            "context": context,
            "error_type": type(error).__name__,
            "message": str(error)
        })

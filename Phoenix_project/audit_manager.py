from typing import Dict, Any, List
import pandas as pd

from .audit.logger import AuditLogger
from .monitor.logging import get_logger
from .core.schemas.fusion_result import FusionResult
from .execution.signal_protocol import StrategySignal

# This is the application logger, distinct from the AuditLogger
logger = get_logger(__name__)

class AuditManager:
    """
    A high-level manager that orchestrates all audit logging.
    
    It provides a simple, centralized interface for other services (like
    Orchestrator, TradeLifecycleManager) to log key events without
    needing to know the underlying logging mechanism (e.g., AuditLogger, S3).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the AuditManager.
        
        Args:
            config (Dict[str, Any]): The main system configuration.
        """
        self.config = config
        
        # Initialize the low-level audit logger
        # This logger writes to the local JSONL file
        self.audit_logger = AuditLogger(config)
        
        # TODO: Initialize other audit destinations (e.g., S3 uploader)
        
        logger.info("AuditManager initialized.")

    def log_event_in(self, event: Dict[str, Any]):
        """Logs a raw event received by the system."""
        try:
            self.audit_logger.log_event_in(event)
        except Exception as e:
            logger.error(f"AuditManager failed to log event_in: {e}")

    def log_pipeline_run(self, fusion_result: FusionResult):
        """
        Logs all relevant data from a single cognitive pipeline run.
        This includes the I/O bundle and the final fused result.
        """
        try:
            # 1. Log the I/O bundle (RAG context, agent prompts/responses)
            if fusion_result.pipeline_io:
                self.audit_logger.log_pipeline_io(
                    fusion_result.event_id, 
                    fusion_result.pipeline_io
                )
                
            # 2. Log the final decision
            # We use model_dump() to get a serializable dict from the Pydantic model
            self.audit_logger.log_fusion_result(fusion_result.model_dump())
            
        except Exception as e:
            logger.error(f"AuditManager failed to log pipeline_run for {fusion_result.event_id}: {e}")

    def log_signal(self, signal: StrategySignal):
        """Logs the generated strategy signal (target weights)."""
        try:
            # Use model_dump() for the Pydantic model
            self.audit_logger.log_signal(signal.model_dump())
        except Exception as e:
            logger.error(f"AuditManager failed to log signal: {e}")

    def log_trade(self, fill_record: Dict[str, Any]):
        """Logs an executed (simulated) trade."""
        try:
            self.audit_logger.log_trade(fill_record)
        except Exception as e:
            logger.error(f"AuditManager failed to log trade: {e}")

    def log_costs(self, costs: Dict[str, float], timestamp: pd.Timestamp):
        """Logs simulated execution costs."""
        try:
            self.audit_logger.log_costs(costs, timestamp)
        except Exception as e:
            logger.error(f"AuditManager failed to log costs: {e}")

    def log_portfolio_snapshot(self, portfolio: Dict[str, float], pnl_record: Dict[str, Any]):
        """Logs a snapshot of the portfolio and PnL."""
        try:
            self.audit_logger.log_portfolio_snapshot(portfolio, pnl_record)
        except Exception as e:
            logger.error(f"AuditManager failed to log portfolio snapshot: {e}")

    def log_system_error(self, error: Exception, context: str = "general"):
        """Logs a critical system error."""
        try:
            self.audit_logger.log_system_error(error, context)
        except Exception as e:
            logger.error(f"CRITICAL: AuditManager failed to log system error: {e}")

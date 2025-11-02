"""
Audit Manager

Provides a high-level interface for logging and retrieving audit data.
This acts as a simplified facade in front of more complex storage
backends (like the CoTDatabase).
"""
import logging
from typing import Optional, Dict, Any
from uuid import UUID

# 修复：从根目录开始使用绝对导入，移除 `.`
from memory.cot_database import CoTDatabase
from core.schemas.data_schema import MarketEvent
from core.schemas.fusion_result import FusionResult

logger = logging.getLogger(__name__)

class AuditManager:
    """
    Facade for handling audit logging to persistent storage.
    """

    def __init__(self, cot_database: CoTDatabase):
        """
        Initializes the AuditManager.

        Args:
            cot_database: An initialized CoTDatabase instance.
        """
        self.db = cot_database
        if self.db is None:
            logger.warning("CoTDatabase is not provided. AuditManager will be disabled.")
        logger.info("AuditManager initialized.")

    async def log_complete_run(self,
                               event: Optional[MarketEvent],
                               task_name: Optional[str],
                               evidence: Dict[str, Any],
                               fusion_result: FusionResult) -> Optional[str]:
        """
        Logs a complete cognitive pipeline run.

        Args:
            event: The triggering MarketEvent (if any).
            task_name: The name of the scheduled task (if any).
            evidence: The dictionary of RAG context.
            fusion_result: The final FusionResult object.

        Returns:
            The unique run_id, or None if logging failed.
        """
        if not self.db:
            logger.warning("Audit logging skipped: CoTDatabase is disabled.")
            return None

        try:
            run_id = await self.db.log_pipeline_run(
                event=event,
                task_name=task_name,
                evidence=evidence,
                fusion_result=fusion_result
            )
            return run_id
        except Exception as e:
            logger.error(f"Failed to log pipeline run to CoTDatabase: {e}", exc_info=True)
            return None

    async def get_run_details(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the full details for a specific pipeline run.

        Args:
            run_id: The unique ID of the run to retrieve.

        Returns:
            A dictionary containing the run details, or None if not found.
        """
        if not self.db:
            logger.warning("Cannot get run details: CoTDatabase is disabled.")
            return None
        
        try:
            # We need to validate if the run_id is a proper UUID
            try:
                # This is just for validation, the DB query uses the string
                _ = UUID(run_id, version=4)
            except ValueError:
                logger.warning(f"Invalid run_id format: {run_id}")
                return None

            # Assuming AuditViewer logic is what we want
            # In a real system, this might call a method on self.db
            # For now, let's assume we need to implement it here or move
            # AuditViewer to be part of the manager.
            
            # This relies on the CoTDatabase having the query methods.
            # Let's proxy the request.
            logger.warning("get_run_details logic is proxied via CoTDatabase. Consider AuditViewer.")
            # This is a conceptual simplification. The AuditViewer class
            # actually has this logic. A better design would be to
            # inject AuditViewer or have self.db provide this.
            
            # Let's assume CoTDatabase *doesn't* have this logic
            # and AuditManager is simplified.
            logger.warning("get_run_details is not fully implemented in AuditManager stub.")
            # A real implementation would query the DB tables.
            return None

        except Exception as e:
            logger.error(f"Error retrieving run details for {run_id}: {e}", exc_info=True)
            return None

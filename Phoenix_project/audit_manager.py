# audit_manager.py

import logging
from datetime import datetime
from ai.tabular_db_client import TabularDBClient  # Assuming a tabular client
from models.evidence import EvidenceItem
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class AuditManager:
    """
    Handles logging of all L1-L3 decisions, final fusion results,
    and associated metadata to a persistent, auditable database.
    """
    def __init__(self, db_client: TabularDBClient):
        self.db_client = db_client
        logger.info("AuditManager initialized.")

    def log_decision_process(
        self,
        decision_id: str,
        ticker: str,
        timestamp: datetime,
        l1_evidence: List[EvidenceItem],
        l2_fusion_result: Dict[str, Any],
        l3_rules_applied: List[str] = None
    ):
        """
        Logs the entire decision-making trail for a single analysis pipeline run.
        """
        logger.info(f"Logging decision trail for ID: {decision_id}")
        
        try:
            decision_data = {
                "decision_id": decision_id,
                "ticker": ticker,
                "timestamp": timestamp,
                "l1_evidence_count": len(l1_evidence),
                "l1_evidence_sources": [item.source for item in l1_evidence],
                "l2_final_score": l2_fusion_result.get("final_posterior_mean"),
                "l2_uncertainty": l2_fusion_result.get("cognitive_uncertainty_score"),
                "l3_rules_applied": l3_rules_applied or [],
                "pnl_result": None  # To be backfilled by TradeLifecycleManager
            }
            
            # 1. Log the main decision record
            self.db_client.insert_record(
                'decision_log',
                decision_data
            )
            
            # 2. Log each piece of L1 evidence individually
            for item in l1_evidence:
                evidence_data = item.model_dump()
                evidence_data["decision_id"] = decision_id
                self.db_client.insert_record(
                    'evidence_log',
                    evidence_data
                )
            
            logger.info(f"Successfully logged decision trail for ID: {decision_id}")

        except Exception as e:
            logger.error(f"Failed to log decision trail for ID {decision_id}: {e}", exc_info=True)

    def update_pnl_for_decision(self, decision_id: str, pnl_result: float):
        """
        Finds the corresponding decision log by decision_id and backfills the pnl_result field.
        """
        logger.info(f"Backfilling P&L for decision_id {decision_id} with result {pnl_result}.")
        try:
            # Assuming the db_client has a method to update records based on a condition.
            # The condition here is finding the record with the matching decision_id.
            self.db_client.update_record(
                'decision_log', {'pnl_result': pnl_result}, {'decision_id': decision_id}
            )
            logger.info(f"Successfully updated P&L for decision_id {decision_id}.")
        except Exception as e:
            logger.error(f"Failed to update P&L for decision_id {decision_id}: {e}", exc_info=True)

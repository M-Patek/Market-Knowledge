# audit_manager.py

import logging
from datetime import datetime
from ai.tabular_db_client import TabularDBClient  # Assuming a tabular client
from ai.validation import EvidenceItem
from execution.interfaces import Order # (L6) Import
from schemas.fusion_result import FusionResult # (L7) Import
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class AuditManager:
    """
    (L7 Patched) Handles logging of all L1-L3 decisions, final fusion results,
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
        input_context: Dict[str, Any], # (L7) e.g., {'prompt': '...', 'data_sources': ['...']}
        l1_evidence: List[EvidenceItem],
        l2_fusion_result: FusionResult, # (L7) Use Pydantic model
        l3_meta_log: Dict[str, Any], # (L7) Use Meta-Log
        output_signal: Dict[str, Any], # (L7) e.g., StrategySignal.model_dump()
        model_versions: Dict[str, str] # (L7) e.g., {'fusion_engine': '1.2', 'l1_analyst': '1.1'}
    ):
        """
        (L7 Patched) Logs the entire decision-making trail for a single analysis pipeline run.
        """
        logger.info(f"Logging decision trail for ID: {decision_id}")
        
        try:
            decision_data = {
                "decision_id": decision_id,
                "ticker": ticker,
                "timestamp": timestamp,
                "model_versions": model_versions, # (L7)
                "input_context": input_context, # (L7)
                "l1_evidence_count": len(l1_evidence),
                "l1_evidence_sources": [item.source for item in l1_evidence],
                "l2_posterior": l2_fusion_result.posterior, # (L7)
                "l2_uncertainty": l2_fusion_result.confidence_interval[1] - l2_fusion_result.confidence_interval[0], # (L7)
                "l3_meta_log": l3_meta_log, # (L7)
                "pnl_result": None,  # To be backfilled by TradeLifecycleManager
                "output_signal": output_signal, # (L7)
            }
            
            # 1. Log the main decision record
            self.db_client.insert_record(
                'decision_log',
                decision_data
            )
            
            # 2. Log each piece of L1 evidence individually
            # (Note: This might be redundant if log_l1_evidence is called separately)
            for item in l1_evidence:
                evidence_data = item.model_dump()
                evidence_data["decision_id"] = decision_id
                evidence_data["model_version"] = model_versions.get(item.source, "N/A") # (L7)
                self.db_client.insert_record(
                    'evidence_log',
                    evidence_data
                )
            
            logger.info(f"Successfully logged decision trail for ID: {decision_id}")

        except Exception as e:
            logger.error(f"Failed to log decision trail for ID {decision_id}: {e}", exc_info=True)

    def log_l1_evidence(self, decision_id: str, l1_results: List[Dict[str, Any]], model_versions: Dict[str, str]):
        """
        (L2 Task 4 / L7 Patched) Logs a batch of L1 agent outputs to the evidence_log.
        Now includes model_version.
        """
        logger.info(f"Logging {len(l1_results)} L1 evidence items for decision_id: {decision_id}")
        try:
            for result in l1_results:
                agent_name = result.get("agent_name", "unknown_agent")
                # Adapt the raw dictionary to the EvidenceItem schema
                evidence_item = EvidenceItem(
                    source=agent_name,
                    content=result.get("analysis", ""),
                    confidence=result.get("confidence", 0.0),
                    timestamp=datetime.now() # Use current time for audit
                )
                evidence_data = evidence_item.model_dump()
                evidence_data["decision_id"] = decision_id
                evidence_data["model_version"] = model_versions.get(agent_name, "N/A") # (L7)
                self.db_client.insert_record('evidence_log', evidence_data)
            logger.info(f"Successfully logged {len(l1_results)} L1 evidence items.")
        except Exception as e:
            logger.error(f"Failed to log L1 evidence batch for ID {decision_id}: {e}", exc_info=True)

    def log_order_submission(self, decision_id: str, order: Order):
        """
        (L6 Task 3) Logs a submitted order to the audit database.
        """
        logger.info(f"Logging order submission for decision_id: {decision_id}")
        try:
            order_data = {
                "decision_id": decision_id,
                "order_id": order.id,
                "ticker": order.ticker,
                "order_type": order.order_type,
                "size": order.size,
                "limit_price": order.limit_price,
                "status": order.status,
                "timestamp": datetime.now()
            }
            self.db_client.insert_record('order_log', order_data)
            logger.info(f"Successfully logged order {order.id} for decision {decision_id}.")

        except Exception as e:
            logger.error(f"Failed to log order submission for decision_id {decision_id}: {e}", exc_info=True)

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

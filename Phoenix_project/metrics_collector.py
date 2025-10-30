import logging
import json
from typing import Dict, Any
from ai.tabular_db_client import TabularDBClient # Assumes db_client is injected

logger = logging.getLogger(__name__)

class MetricsCollector:
    """
    (L7 Task 4) Periodically aggregates metrics from all layers by querying
    the audit database. Outputs a JSON blob for a dashboard.
    """
    def __init__(self, db_client: TabularDBClient):
        self.db_client = db_client
        logger.info("MetricsCollector initialized.")

    def collect_dashboard_metrics(self) -> str:
        """
        Aggregates metrics from the database and returns a JSON string.
        """
        logger.info("Collecting dashboard metrics...")
        try:
            # --- Placeholder Queries ---
            # In a real system, these would be efficient SQL 'GROUP BY' queries.
            # We are simulating the result of those queries here.
            
            # 1. Get recent decisions
            recent_decisions = self.db_client.get_all_records('decision_log') # Placeholder
            
            # 2. Aggregate P&L and Uncertainty
            total_decisions = len(recent_decisions)
            profitable_decisions = 0
            total_pnl = 0.0
            uncertainty_scores = []
            conflict_counts = []
            
            for decision in recent_decisions:
                pnl = decision.get('pnl_result')
                if pnl:
                    total_pnl += pnl
                    if pnl > 0:
                        profitable_decisions += 1
                
                uncertainty = decision.get('l2_uncertainty')
                if uncertainty:
                    uncertainty_scores.append(uncertainty)
                
                meta_log = decision.get('l3_meta_log', {})
                # (L7) We need to ensure 'conflict_count' is logged to the meta_log
                # Let's assume it's logged from the FusionResult.conflict_log
                if 'conflict_log' in decision: 
                    conflict_counts.append(len(decision['conflict_log']))

            # 3. Aggregate L1 Agent Stats (from 'evidence_log')
            all_evidence = self.db_client.get_all_records('evidence_log') # Placeholder
            agent_counts = {}
            for ev in all_evidence:
                source = ev.get('source')
                if source:
                    agent_counts[source] = agent_counts.get(source, 0) + 1

            # 4. Build the JSON output
            dashboard_json = {
                "summary": {
                    "total_decisions": total_decisions,
                    "total_pnl": round(total_pnl, 2),
                    "win_rate": round(profitable_decisions / total_decisions, 2) if total_decisions > 0 else 0,
                    "avg_uncertainty": round(sum(uncertainty_scores) / len(uncertainty_scores), 2) if uncertainty_scores else 0,
                    "avg_conflicts": round(sum(conflict_counts) / len(conflict_counts), 2) if conflict_counts else 0,
                },
                "agent_activity": agent_counts
            }
            
            logger.info(f"Successfully collected dashboard metrics: {dashboard_json['summary']}")
            return json.dumps(dashboard_json, indent=4)

        except Exception as e:
            logger.error(f"Failed to collect dashboard metrics: {e}", exc_info=True)
            return json.dumps({"error": str(e)})

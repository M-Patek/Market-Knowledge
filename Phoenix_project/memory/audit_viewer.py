"""
Audit Viewer
A simple CLI or web tool to query and view audit logs
stored in the CoTDatabase (PostgreSQL).
"""
import logging
from datetime import datetime, timedelta
import pandas as pd
from typing import Optional

# 修复：将相对导入 'from .cot_database...' 更改为绝对导入
from memory.cot_database import CoTDatabase

logger = logging.getLogger(__name__)

class AuditViewer:
    """
    Provides methods to query the audit database.
    """

    def __init__(self, cot_database: CoTDatabase):
        """
        Initializes the AuditViewer.

        Args:
            cot_database: An initialized CoTDatabase instance.
        """
        self.db = cot_database
        logger.info("AuditViewer initialized.")

    async def get_recent_runs(self, limit: int = 10) -> Optional[pd.DataFrame]:
        """
        Retrieves the most recent pipeline runs.
        """
        query = """
        SELECT run_id, run_timestamp, event_id, task_name
        FROM pipeline_runs
        ORDER BY run_timestamp DESC
        LIMIT $1
        """
        try:
            records = await self.db.fetch(query, limit)
            if records:
                return pd.DataFrame(records, columns=['run_id', 'run_timestamp', 'event_id', 'task_name'])
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching recent runs: {e}", exc_info=True)
            return None

    async def get_run_details(self, run_id: str) -> Optional[dict]:
        """
        Retrieves all details for a specific run_id, including
        the event, evidence, and fusion result.
        """
        details = {}
        
        try:
            # 1. Get the main run
            run_query = "SELECT * FROM pipeline_runs WHERE run_id = $1"
            run_data = await self.db.fetchrow(run_query, run_id)
            if not run_data:
                logger.warning(f"No run found with run_id: {run_id}")
                return None
            details['run_info'] = dict(run_data)
            
            # 2. Get the triggering event
            event_id = run_data['event_id']
            if event_id:
                event_query = "SELECT * FROM market_events WHERE event_id = $1"
                event_data = await self.db.fetchrow(event_query, event_id)
                details['triggering_event'] = dict(event_data) if event_data else "Event ID not found"
                
            # 3. Get the evidence context
            evidence_query = "SELECT * FROM evidence_context WHERE run_id = $1"
            evidence_data = await self.db.fetchrow(evidence_query, run_id)
            details['evidence_context'] = dict(evidence_data) if evidence_data else "No evidence found"

            # 4. Get the fusion result
            fusion_query = "SELECT * FROM fusion_results WHERE run_id = $1"
            fusion_data = await self.db.fetchrow(fusion_query, run_id)
            details['fusion_result'] = dict(fusion_data) if fusion_data else "No fusion result found"

            return details
            
        except Exception as e:
            logger.error(f"Error fetching details for run_id {run_id}: {e}", exc_info=True)
            return None

# Example CLI Usage
if __name__ == "__main__":
    import asyncio
    import os
    import json
    
    logging.basicConfig(level=logging.INFO)
    
    # Requires a running PostgreSQL DB defined in env vars
    DB_URL = os.environ.get("POSTGRES_URL")
    
    if not DB_URL:
        logger.error("POSTGRES_URL environment variable not set. Cannot run example.")
    else:
        async def main():
            db = CoTDatabase(database_url=DB_URL)
            await db.connect()
            
            viewer = AuditViewer(cot_database=db)
            
            print("--- Fetching recent runs ---")
            recent_runs = await viewer.get_recent_runs(limit=5)
            if recent_runs is not None:
                print(recent_runs.to_string())
                
                if not recent_runs.empty:
                    # Get details for the most recent run
                    latest_run_id = recent_runs.iloc[0]['run_id']
                    print(f"\n--- Fetching details for run_id: {latest_run_id} ---")
                    
                    details = await viewer.get_run_details(latest_run_id)
                    if details:
                        # Use pandas to pretty-print JSON-like dicts
                        print(json.dumps(details, indent=2, default=str))
            
            await db.disconnect()

        asyncio.run(main())

import sqlite3
from typing import Dict, Any, List, Optional
import json

from ..monitor.logging import get_logger

logger = get_logger(__name__)

class CoTDatabase:
    """
    Manages a local SQLite database for storing "Chains of Thought" (CoT)
    and other complex reasoning artifacts from the AI pipeline.
    
    This provides a persistent, queryable, and auditable record of *how*
    the AI reached its conclusions, which is more structured than
    a simple JSONL audit log.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the CoTDatabase.
        
        Args:
            config: Main system configuration. Expects 'cot_database.db_path'.
        """
        db_config = config.get('cot_database', {})
        self.db_path = db_config.get('db_path', 'logs/cot_audit.db')
        self.conn = None
        logger.info(f"CoTDatabase initialized at: {self.db_path}")

    def connect(self):
        """Establishes the SQLite connection and creates tables."""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row # Access results by column name
            self._create_tables()
            logger.info(f"CoTDatabase connected and tables verified.")
        except Exception as e:
            logger.error(f"Failed to connect to CoTDatabase: {e}", exc_info=True)
            raise

    def close(self):
        """Closes the SQLite connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("CoTDatabase connection closed.")
            
    def _create_tables(self):
        """Creates the necessary tables if they don't exist."""
        with self.conn:
            # Main table for each pipeline run
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                event_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                headline TEXT,
                final_decision TEXT,
                confidence REAL,
                cognitive_uncertainty REAL,
                status TEXT
            );
            """)
            
            # Table for individual agent decisions in a run
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_decisions (
                decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                decision TEXT,
                confidence REAL,
                justification TEXT,
                metadata_json TEXT,
                FOREIGN KEY (event_id) REFERENCES pipeline_runs (event_id)
            );
            """)
            
            # Table for the raw RAG context used
            self.conn.execute("""
            CREATE TABLE IF NOT EXISTS retrieval_context (
                context_id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT NOT NULL,
                retrieval_type TEXT, -- e.g., 'vector', 'temporal'
                context_json TEXT,
                FOREIGN KEY (event_id) REFERENCES pipeline_runs (event_id)
            );
            """)

    def log_pipeline_run(self, fusion_result: 'FusionResult'):
        """
        Logs a complete pipeline run (FusionResult) to the database
        in a transactional manner.
        
        Args:
            fusion_result (FusionResult): The Pydantic model output.
        """
        if not self.conn:
            logger.error("Database not connected. Cannot log pipeline run.")
            return

        try:
            with self.conn:
                # 1. Insert into main 'pipeline_runs' table
                self.conn.execute(
                    """
                    INSERT INTO pipeline_runs 
                        (event_id, timestamp, headline, final_decision, confidence, cognitive_uncertainty, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(event_id) DO UPDATE SET
                        timestamp = excluded.timestamp,
                        headline = excluded.headline,
                        final_decision = excluded.final_decision,
                        confidence = excluded.confidence,
                        cognitive_uncertainty = excluded.cognitive_uncertainty,
                        status = excluded.status
                    """,
                    (
                        fusion_result.event_id,
                        fusion_result.pipeline_io.get('event_data', {}).get('timestamp', pd.Timestamp.now().isoformat()),
                        fusion_result.pipeline_io.get('event_data', {}).get('headline', ''),
                        fusion_result.final_decision.decision,
                        fusion_result.final_decision.confidence,
                        fusion_result.cognitive_uncertainty,
                        fusion_result.status
                    )
                )
                
                # 2. Delete old agent decisions for this event_id to avoid duplicates
                self.conn.execute("DELETE FROM agent_decisions WHERE event_id = ?", (fusion_result.event_id,))
                
                # 3. Insert new agent decisions
                for decision in fusion_result.agent_decisions:
                    self.conn.execute(
                        """
                        INSERT INTO agent_decisions
                            (event_id, agent_id, decision, confidence, justification, metadata_json)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            fusion_result.event_id,
                            decision.agent_id,
                            decision.decision,
                            decision.confidence,
                            decision.justification,
                            json.dumps(decision.metadata)
                        )
                    )
                
                # TODO: Add logic to log retrieval_context
                
        except Exception as e:
            logger.error(f"Failed to log pipeline run {fusion_result.event_id} to CoTDatabase: {e}", exc_info=True)

    def get_run_by_event_id(self, event_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a full pipeline run, including all agent decisions.
        
        Args:
            event_id (str): The event ID.
            
        Returns:
            Optional[Dict]: A consolidated dictionary, or None if not found.
        """
        if not self.conn:
            logger.error("Database not connected.")
            return None
            
        try:
            # 1. Get main run
            run_data = self.conn.execute("SELECT * FROM pipeline_runs WHERE event_id = ?", (event_id,)).fetchone()
            if not run_data:
                return None
                
            run_result = dict(run_data)
            
            # 2. Get agent decisions
            agent_decisions_raw = self.conn.execute("SELECT * FROM agent_decisions WHERE event_id = ?", (event_id,)).fetchall()
            
            agent_decisions = []
            for row in agent_decisions_raw:
                decision = dict(row)
                decision['metadata'] = json.loads(decision.pop('metadata_json', '{}'))
                agent_decisions.append(decision)
                
            run_result['agent_decisions'] = agent_decisions
            
            # TODO: Add logic to retrieve retrieval_context
            
            return run_result
            
        except Exception as e:
            logger.error(f"Failed to retrieve run {event_id} from CoTDatabase: {e}", exc_info=True)
            return None

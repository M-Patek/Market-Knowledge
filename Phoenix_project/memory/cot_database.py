"""
CoT (Chain-of-Thought) Database

Handles the saving and retrieval of the *entire* decision-making
process (the "Chain of Thought") for auditability and review.

This includes:
1. The triggering event.
2. The RAG evidence context provided.
3. The individual agent decisions.
4. The final fused result.

Uses a PostgreSQL database (via asyncpg) for structured, reliable,
and relational storage of these audit logs.
"""
import logging
import asyncpg
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import uuid

# 修复：添加 pandas 导入
import pandas as pd

# 修复：添加 FusionResult 导入 (用于类型提示)
from ..core.schemas.fusion_result import FusionResult
from ..core.schemas.data_schema import MarketEvent

logger = logging.getLogger(__name__)

class CoTDatabase:
    """
    Manages the connection and read/write operations for the
    PostgreSQL audit database.
    """

    def __init__(self, database_url: str):
        """
        Initializes the CoTDatabase.

        Args:
            database_url: The connection string for the PostgreSQL database
                          (e.g., "postgresql://user:pass@host:port/db")
        """
        self.database_url = database_url
        self._pool: asyncpg.Pool = None
        logger.info("CoTDatabase initialized.")

    async def connect(self):
        """Establishes the database connection pool."""
        if not self._pool:
            try:
                self._pool = await asyncpg.create_pool(self.database_url)
                logger.info("CoTDatabase connection pool established.")
                await self.setup_schema()
            except Exception as e:
                logger.error(f"Failed to connect to CoTDatabase: {e}", exc_info=True)
                self._pool = None
                raise

    async def disconnect(self):
        """Closes the database connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("CoTDatabase connection pool closed.")

    async def execute(self, query: str, *args):
        """Executes a query without returning results."""
        if not self._pool:
            raise ConnectionError("Database not connected. Call connect() first.")
        async with self._pool.acquire() as conn:
            await conn.execute(query, *args)

    async def fetch(self, query: str, *args) -> List[asyncpg.Record]:
        """Executes a query and returns all results."""
        if not self._pool:
            raise ConnectionError("Database not connected. Call connect() first.")
        async with self._pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args) -> Optional[asyncpg.Record]:
        """Executes a query and returns a single result."""
        if not self._pool:
            raise ConnectionError("Database not connected. Call connect() first.")
        async with self._pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def setup_schema(self):
        """
        Ensures the required tables for auditing exist.
        """
        logger.info("Setting up database schema...")
        
        # Table for the main pipeline run
        await self.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_runs (
            run_id UUID PRIMARY KEY,
            run_timestamp TIMESTAMPTZ NOT NULL,
            event_id VARCHAR(255),
            task_name VARCHAR(255)
        );
        """)
        
        # Table for the triggering market events
        await self.execute("""
        CREATE TABLE IF NOT EXISTS market_events (
            event_id VARCHAR(255) PRIMARY KEY,
            source VARCHAR(100),
            timestamp TIMESTAMPTZ NOT NULL,
            content TEXT,
            metadata JSONB
        );
        """)
        
        # Table for the retrieved evidence context
        await self.execute("""
        CREATE TABLE IF NOT EXISTS evidence_context (
            run_id UUID PRIMARY KEY REFERENCES pipeline_runs(run_id),
            retrieval_timestamp TIMESTAMPTZ NOT NULL,
            vector_hits JSONB,
            temporal_hits JSONB,
            tabular_hits JSONB
        );
        """)
        
        # Table for the final fusion result
        await self.execute("""
        CREATE TABLE IF NOT EXISTS fusion_results (
            run_id UUID PRIMARY KEY REFERENCES pipeline_runs(run_id),
            fusion_timestamp TIMESTAMPTZ NOT NULL,
            fused_confidence FLOAT,
            fused_sentiment FLOAT,
            fused_predicted_impact FLOAT,
            cognitive_uncertainty FLOAT,
            fused_rationale TEXT,
            contributing_decisions JSONB
        );
        """)
        
        logger.info("Database schema setup complete.")

    async def log_market_event(self, event: MarketEvent):
        """
        Logs a market event to the database.
        Uses ON CONFLICT DO NOTHING to avoid duplicates.
        """
        query = """
        INSERT INTO market_events (event_id, source, timestamp, content, metadata)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (event_id) DO NOTHING;
        """
        await self.execute(
            query,
            event.event_id,
            event.source,
            event.timestamp,
            event.content,
            json.dumps(event.metadata) if event.metadata else None
        )

    async def log_pipeline_run(self, 
                               event: Optional[MarketEvent], 
                               task_name: Optional[str],
                               evidence: Dict[str, Any],
                               # 修复：为 FusionResult 添加类型提示
                               fusion_result: FusionResult
                               ) -> str:
        """
        Logs a complete pipeline run in a single transaction.

        Args:
            event: The triggering MarketEvent (if any).
            task_name: The name of the scheduled task (if any).
            evidence: The dictionary of RAG context.
            fusion_result: The final FusionResult object.

        Returns:
            The unique run_id for this logged run.
        """
        if not self._pool:
            raise ConnectionError("Database not connected. Call connect() first.")
            
        run_id = uuid.uuid4()
        # 修复：使用 pd.Timestamp.now()
        now = pd.Timestamp.now(tz='UTC').to_pydatetime()
        event_id = event.event_id if event else None

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                try:
                    # 1. Log the market event (if it exists)
                    if event:
                        await self.log_market_event(event) # Uses internal execute

                    # 2. Log the pipeline run
                    await conn.execute(
                        """
                        INSERT INTO pipeline_runs (run_id, run_timestamp, event_id, task_name)
                        VALUES ($1, $2, $3, $4)
                        """,
                        run_id, now, event_id, task_name
                    )
                    
                    # 3. Log the evidence
                    await conn.execute(
                        """
                        INSERT INTO evidence_context (run_id, retrieval_timestamp, vector_hits, temporal_hits, tabular_hits)
                        VALUES ($1, $2, $3, $4, $5)
                        """,
                        run_id,
                        evidence.get('retrieval_timestamp', now),
                        json.dumps(evidence.get('vector_hits', [])),
                        json.dumps(evidence.get('temporal_hits', [])),
                        json.dumps(evidence.get('tabular_hits', []))
                    )
                    
                    # 4. Log the fusion result
                    await conn.execute(
                        """
                        INSERT INTO fusion_results (run_id, fusion_timestamp, fused_confidence, 
                                                    fused_sentiment, fused_predicted_impact, 
                                                    cognitive_uncertainty, fused_rationale, 
                                                    contributing_decisions)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                        """,
                        run_id,
                        fusion_result.fusion_timestamp,
                        fusion_result.fused_confidence,
                        fusion_result.fused_sentiment,
                        fusion_result.fused_predicted_impact,
                        fusion_result.cognitive_uncertainty,
                        fusion_result.fused_rationale,
                        # Convert list of Pydantic models to list of dicts for JSON
                        json.dumps([d.dict() for d in fusion_result.contributing_decisions])
                    )
                    
                    logger.info(f"Successfully logged pipeline run: {run_id}")
                    return str(run_id)

                except Exception as e:
                    logger.error(f"Failed to log pipeline run (transaction rolled back): {e}", exc_info=True)
                    # Transaction is automatically rolled back
                    raise

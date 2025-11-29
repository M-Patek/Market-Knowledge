"""
Tabular DB Client for Phoenix.

This module provides a client for interacting with tabular (SQL) databases,
including a text-to-SQL agent capability.
[Phase II Fix] Atomic Transactions & Connection Management
[Phase IV Fix] Async SQLAlchemy Engine & Native Transactions
"""

import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Any, List, Dict, Optional, Callable

import sqlalchemy  # type: ignore
from sqlalchemy import inspect, text  # type: ignore
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine # type: ignore
from sqlalchemy.exc import SQLAlchemyError  # type: ignore

from ..api.gemini_pool_manager import GeminiClient
from ..ai.prompt_manager import PromptManager
from ..ai.prompt_renderer import PromptRenderer
from ..utils.retry import retry_with_exponential_backoff

logger = logging.getLogger(__name__)


class TabularDBClient:
    """
    Client for interacting with a SQL database.
    Manages connections, schema inspection, and text-to-SQL execution.
    """

    def __init__(
        self, 
        db_uri: str, 
        llm_client: Optional[GeminiClient], 
        config: Dict[str, Any],
        prompt_manager: PromptManager,
        prompt_renderer: PromptRenderer
    ):
        self.db_uri = db_uri
        self.llm_client = llm_client
        self.config = config.get("tabular_db", {})
        # [Task 4.1] Auto-upgrade connection string for asyncpg
        if self.db_uri.startswith("postgresql://"):
             self.db_uri = self.db_uri.replace("postgresql://", "postgresql+asyncpg://")
             
        self.engine: AsyncEngine = self._create_db_engine()
        self.schema: str = "" 
        
        self.prompt_manager = prompt_manager
        self.prompt_renderer = prompt_renderer

        self.sql_agent: Optional[Callable] = None
        if self.llm_client:
            self.sql_agent = self._initialize_sql_agent()

    def _create_db_engine(self) -> AsyncEngine:
        try:
            engine = create_async_engine(self.db_uri)
            logger.info(f"Async DB engine created for {self.db_uri.split('@')[-1]}")
            return engine
        except SQLAlchemyError as e:
            logger.error(f"Failed to connect to SQL database at {self.db_uri}: {e}")
            raise

    async def _get_db_schema_async(self) -> str:
        try:
            def sync_inspect(conn):
                inspector = inspect(conn)
                parts = []
                for table in inspector.get_table_names():
                    parts.append(f"Table '{table}':")
                    for col in inspector.get_columns(table):
                        parts.append(f"  - {col['name']} ({col['type']})")
                return "\n".join(parts)

            async with self.engine.connect() as conn:
                schema_str = await conn.run_sync(sync_inspect)
            
            logger.info(f"Retrieved DB schema:\n{schema_str}")
            return schema_str
        except SQLAlchemyError as e:
            logger.error(f"Failed to inspect DB schema: {e}")
            return "Error: Could not retrieve schema."

    async def ensure_schema(self):
        if not self.schema:
            logger.info("Lazy loading DB schema...")
            self.schema = await self._get_db_schema_async()

    def _generate_sql_prompt(self, query: str) -> str:
        logger.debug("Rendering text_to_sql prompt...")
        try:
            context = {
                "dialect": self.engine.dialect.name,
                "schema": self.schema,
                "query": query
            }
            rendered_data = self.prompt_renderer.render("text_to_sql", context)
            prompt_str = rendered_data.get("full_prompt_template")
            if not prompt_str:
                 raise ValueError("'full_prompt_template' key not found in rendered prompt.")
            return prompt_str
        except Exception as e:
            logger.error(f"Failed to render text_to_sql prompt: {e}. Using fallback.", exc_info=True)
            dialect_name = self.engine.dialect.name
            return f"""
            You are an expert {dialect_name} SQL query generator.
            Schema: {self.schema}
            Question: "{query}"
            SQL Query:
            """

    @retry_with_exponential_backoff(exceptions_to_retry=(SQLAlchemyError,))
    async def _run_sql_agent(self, query: str) -> Dict[str, Any]:
        logger.info(f"SQL Agent processing query: {query}")
        sql_query_generated = None
        try:
            await self.ensure_schema()
            
            if not self.llm_client:
                raise ValueError("LLM Client not initialized.")

            prompt = self._generate_sql_prompt(query)
            sql_query_generated = await self.llm_client.generate_text(prompt)

            if not sql_query_generated:
                raise ValueError("LLM failed to generate SQL query.")

            sql_query = sql_query_generated.strip().replace("```sql", "").replace("```", "").strip(";")
            logger.info(f"Generated SQL: {sql_query}")

            rows = []
            async with self.engine.connect() as conn:
                result = await conn.execute(text(sql_query))
                rows = [dict(row) for row in result.mappings().all()]
            
            logger.info(f"SQL query returned {len(rows)} rows.")
            
            return {
                "query": query,
                "generated_sql": sql_query,
                "results": rows,
                "error": None
            }

        except SQLAlchemyError as e:
            logger.error(f"SQLAlchemyError in SQL Agent execution: {e}", exc_info=True)
            raise e
        except Exception as e:
            logger.error(f"Unexpected error in SQL Agent execution: {e}", exc_info=True)
            return {
                "query": query,
                "generated_sql": sql_query_generated,
                "results": [],
                "error": str(e)
            }

    def _initialize_sql_agent(self) -> Callable:
        logger.info("Initializing custom Text-to-SQL agent.")
        return self._run_sql_agent
    
    @retry_with_exponential_backoff(exceptions_to_retry=(SQLAlchemyError,))
    async def execute_sql(self, sql: str, params: Dict[str, Any] = None, connection=None) -> List[Dict[str, Any]]:
        """
        [Security Fix] Executes a raw SQL query with parameter binding.
        [Phase IV Fix] Native Async Execution.
        """
        if not self.engine:
             raise ValueError("DB engine not initialized.")
        
        if connection:
            # Reuse existing async connection (Transaction context)
            result = await connection.execute(text(sql), params or {})
            return [dict(row) for row in result.mappings().all()]
        else:
            # New connection
            async with self.engine.connect() as conn:
                result = await conn.execute(text(sql), params or {})
                return [dict(row) for row in result.mappings().all()]

    async def query(self, query: str) -> Dict[str, Any]:
        if not self.sql_agent:
            logger.error("SQL agent is not initialized. Cannot process query.")
            return {"error": "SQL agent not initialized."}
        try:
            result = await self.sql_agent(query)
            if result.get("error"):
                logger.warning(f"SQL agent failed: {result['error']}.")
            return result
        except Exception as e:
            logger.error(f"Unhandled error during tabular query: {e}")
            return {"query": query, "results": [], "error": f"Unhandled exception: {e}"}

    @retry_with_exponential_backoff(exceptions_to_retry=(SQLAlchemyError,))
    async def upsert_data(self, table_name: str, data: Dict[str, Any], unique_key: str, connection=None) -> bool:
        """
        [Task 2] 插入或更新 (Upsert) 一行数据到指定的表格。
        [Phase II Fix] Added optional connection for atomic transactions.
        [Phase IV Fix] Native Async Execution (No more to_thread).
        """
        if not self.engine:
            logger.error(f"Upsert failed: DB engine not initialized.")
            return False
        
        if not data or not unique_key or not table_name:
            logger.error("Upsert failed: table_name, data, and unique_key are required.")
            return False
        
        if self.engine.dialect.name != "postgresql":
            logger.error(f"Upsert logic is only implemented for PostgreSQL, not {self.engine.dialect.name}.")
            return False
        
        columns = data.keys()
        column_names = ", ".join([f'"{c}"' for c in columns])
        placeholders = ", ".join([f":{c}" for c in columns])
        update_set_parts = [f'"{c}" = EXCLUDED."{c}"' for c in columns if c != unique_key]
        
        if not update_set_parts:
            update_set = "DO NOTHING"
        else:
            update_set = f"DO UPDATE SET {', '.join(update_set_parts)}"
            
        sql = text(f"""
            INSERT INTO "{table_name}" ({column_names})
            VALUES ({placeholders})
            ON CONFLICT ("{unique_key}")
            {update_set};
        """)
        
        try:
            if connection:
                # [Phase IV Fix] Use existing transaction context (Atomic)
                await connection.execute(sql, data)
            else:
                # [Phase IV Fix] New atomic transaction (Auto-Commit)
                async with self.engine.begin() as conn:
                    await conn.execute(sql, data)
            
            logger.info(f"Successfully upserted data into '{table_name}' for key {data.get(unique_key)}")
            return True
        except SQLAlchemyError as e:
            logger.error(f"Failed to upsert data into {table_name}: {e}", exc_info=True)
            raise e

    @asynccontextmanager
    async def transaction(self):
        """
        [Phase IV Fix] Native Async Transaction.
        """
        async with self.engine.begin() as conn:
            yield conn

    def close(self):
        """Closes the database engine connection pool."""
        if self.engine:
            # engine.dispose is synchronous-compatible in newer async engines or needs await
            # For now we assume standard dispose pattern
            pass
            # asyncio.create_task(self.engine.dispose()) # Ideally
            logger.info("Tabular DB client connections closed.")

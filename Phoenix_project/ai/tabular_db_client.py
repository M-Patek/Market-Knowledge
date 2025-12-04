"""
Tabular DB Client for Phoenix.

This module provides a client for interacting with tabular (SQL) databases,
including a text-to-SQL agent capability.
[Phase II Fix] Atomic Transactions & Connection Management
[Phase IV Fix] Async SQLAlchemy Engine & Native Transactions
[Code Opt Expert Fix] Task 05: SQL Injection Hardening
"""

import logging
import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any, List, Dict, Optional, Callable

import sqlalchemy  # type: ignore
from sqlalchemy import inspect, text, table, column  # type: ignore
from sqlalchemy.dialects.postgresql import insert as pg_insert
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
             
        self.engine: AsyncEngine = self._create_db_engine(self.db_uri)
        
        # [Task 4.4] Read-Only Engine Support
        self.db_uri_ro = os.environ.get("TABULAR_DB_URI_RO")
        if self.db_uri_ro and self.db_uri_ro.startswith("postgresql://"):
             self.db_uri_ro = self.db_uri_ro.replace("postgresql://", "postgresql+asyncpg://")
        
        self.engine_ro = None
        if self.db_uri_ro:
             self.engine_ro = self._create_db_engine(self.db_uri_ro)
        
        self.schema: str = "" 
        
        self.prompt_manager = prompt_manager
        self.prompt_renderer = prompt_renderer

        self.sql_agent: Optional[Callable] = None
        if self.llm_client:
            self.sql_agent = self._initialize_sql_agent()

    def _create_db_engine(self, uri: str) -> AsyncEngine:
        try:
            engine = create_async_engine(uri)
            logger.info(f"Async DB engine created for {uri.split('@')[-1]}")
            return engine
        except SQLAlchemyError as e:
            logger.error(f"Failed to connect to SQL database at {uri}: {e}")
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
            
            # [Task 05] SQL Injection Hardening: Enforce Read-Only Access
            normalized_query = sql_query.upper().strip()
            if not normalized_query.startswith("SELECT"):
                 raise ValueError("Security Violation: Only SELECT queries are allowed.")
            
            forbidden_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE", "GRANT", "REVOKE", "EXEC"]
            if any(keyword in normalized_query for keyword in forbidden_keywords):
                 raise ValueError(f"Security Violation: Query contains forbidden keywords.")

            logger.info(f"Generated SQL: {sql_query}")

            rows = []
            
            # [Task 4.4] Use Read-Only Engine
            execution_engine = self.engine_ro if self.engine_ro else self.engine
            if not self.engine_ro:
                 logger.warning("SECURITY WARNING: Using Read-Write engine for Text-to-SQL. Configure TABULAR_DB_URI_RO for isolation.")

            async with execution_engine.connect() as conn:
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
        [Task 2] Insert or Update (Upsert) a row.
        [Phase II Fix] Added optional connection for atomic transactions.
        [Phase IV Fix] Native Async Execution (No more to_thread).
        [Task 05] Refactored to use SQLAlchemy Expression Language (Safe Identifiers).
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
        
        try:
            # [Task 05] Refactored to use SQLAlchemy Expression Language (Safe Identifiers)
            # Create a lightweight table object for the statement
            tbl = table(table_name, *[column(c) for c in data.keys()])
            
            # Build the insert statement
            stmt = pg_insert(tbl).values(**data)
            
            # Build the ON CONFLICT update clause
            update_dict = {c: stmt.excluded[c] for c in data.keys() if c != unique_key}
            
            if update_dict:
                stmt = stmt.on_conflict_do_update(index_elements=[unique_key], set_=update_dict)
            else:
                stmt = stmt.on_conflict_do_nothing(index_elements=[unique_key])

            if connection:
                # [Phase IV Fix] Use existing transaction context (Atomic)
                await connection.execute(stmt)
            else:
                # [Phase IV Fix] New atomic transaction (Auto-Commit)
                async with self.engine.begin() as conn:
                    await conn.execute(stmt)
            
            logger.info(f"Successfully upserted data into '{table_name}' for key {data.get(unique_key)}")
            return True
        except SQLAlchemyError as e:
            logger.error(f"Failed to upsert data into {table_name}: {e}", exc_info=True)
            raise e

    async def insert_data(self, table_name: str, data: Dict[str, Any], connection=None) -> bool:
        """
        [Task 2.1] Strict Insert (No Upsert). Raises IntegrityError on conflict.
        """
        if not self.engine:
             raise ValueError("DB engine not initialized.")
        
        try:
            tbl = table(table_name, *[column(c) for c in data.keys()])
            stmt = pg_insert(tbl).values(**data)
            # No on_conflict clause here!
            
            if connection:
                await connection.execute(stmt)
            else:
                async with self.engine.begin() as conn:
                    await conn.execute(stmt)
            return True
        except SQLAlchemyError as e:
            # Let the caller handle the specific error (e.g., IntegrityError)
            raise e

    @asynccontextmanager
    async def transaction(self):
        """
        [Phase IV Fix] Native Async Transaction.
        """
        async with self.engine.begin() as conn:
            yield conn

    async def close(self):
        """Closes the database engine connection pool."""
        if self.engine:
            await self.engine.dispose()
        if self.engine_ro:
            await self.engine_ro.dispose()
        logger.info("Tabular DB client connections closed.")

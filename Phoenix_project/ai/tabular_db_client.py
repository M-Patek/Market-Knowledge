"""
Tabular DB Client for Phoenix.

This module provides a client for interacting with tabular (SQL) databases,
including a text-to-SQL agent capability.
"""

import logging
import re
from typing import Any, List, Dict, Optional, Callable

import sqlalchemy  # type: ignore
from sqlalchemy import create_engine, inspect, text, Engine  # type: ignore
from sqlalchemy.exc import SQLAlchemyError  # type: ignore

# 这是一个 LangChain 的特定导入, 我们将尝试用一个
# 适应项目自身 LLMClient 的自定义实现来替换它
# from langchain_community.agent_toolkits import create_sql_agent
# from langchain_community.utilities.sql_database import SQLDatabase

from ..api.gemini_pool_manager import GeminiClient

logger = logging.getLogger(__name__)


class TabularDBClient:
    """
    Client for interacting with a SQL database.
    Manages connections, schema inspection, and text-to-SQL execution.
    """

    def __init__(
        self, db_uri: str, llm_client: GeminiClient, config: Dict[str, Any]
    ):
        """
        Initializes the TabularDBClient.

        Args:
            db_uri: The SQLAlchemy connection string (e.g., "postgresql://user:pass@host/db").
            llm_client: An instance of the GeminiClient.
            config: Configuration dictionary.
        """
        self.db_uri = db_uri
        self.llm_client = llm_client
        self.config = config.get("tabular_db", {})
        self.engine: Engine = self._create_db_engine()
        self.schema: str = self._get_db_schema()

        # 优化: 替换 _mock_sql_agent
        self.sql_agent: Callable = self._initialize_sql_agent()

    def _create_db_engine(self) -> Engine:
        """Creates the SQLAlchemy engine."""
        try:
            engine = create_engine(self.db_uri)
            # 测试连接
            with engine.connect() as connection:
                logger.info(
                    f"Successfully connected to tabular DB: {engine.url.database}"
                )
            return engine
        except SQLAlchemyError as e:
            logger.error(f"Failed to connect to SQL database at {self.db_uri}: {e}")
            raise

    def _get_db_schema(self) -> str:
        """Inspects the database and retrieves the schema as a string."""
        try:
            inspector = inspect(self.engine)
            schema_str_parts = []
            tables = inspector.get_table_names()
            for table in tables:
                schema_str_parts.append(f"Table '{table}':")
                columns = inspector.get_columns(table)
                for col in columns:
                    schema_str_parts.append(
                        f"  - {col['name']} ({col['type']})"
                    )
            
            schema_str = "\n".join(schema_str_parts)
            logger.info(f"Retrieved DB schema:\n{schema_str}")
            return schema_str
        except SQLAlchemyError as e:
            logger.error(f"Failed to inspect DB schema: {e}")
            return "Error: Could not retrieve schema."

    def _generate_sql_prompt(self, query: str) -> str:
        """Generates a prompt for the LLM to convert text to SQL."""
        # 优化：这个模板应该从 PromptManager 加载
        # dialect.name 提供了 SQL 方言 (例如, postgresql)
        return f"""
        You are an expert {self.engine.dialect.name} SQL query generator.
        Your task is to convert a natural language question into a SQL query
        based on the provided database schema.

        Database Schema:
        {self.schema}

        Rules:
        1. Only generate the SQL query. No preamble, no explanation.
        2. The query must be syntactically correct for {self.engine.dialect.name}.
        3. Only query tables and columns present in the schema.
        4. Be careful with data types (e.g., casting).
        5. If the question is complex, break it down (e.g., using WITH clauses).
        6. The query should be read-only (SELECT statements only).
        7. Do NOT generate any INSERT, UPDATE, DELETE, or DROP statements.

        Question:
        "{query}"

        SQL Query:
        """

    async def _run_sql_agent(self, query: str) -> Dict[str, Any]:
        """
        OPTIMIZED: This is the actual agent execution logic.
        It generates SQL from text, executes it, and returns the result.
        """
        logger.info(f"SQL Agent processing query: {query}")
        try:
            # 1. 生成 SQL
            prompt = self._generate_sql_prompt(query)
            # 假设 llm_client.generate_text 是一个异步方法
            sql_query = await self.llm_client.generate_text(prompt)

            if not sql_query:
                raise ValueError("LLM failed to generate SQL query.")

            # 2. 清理和验证 SQL
            sql_query = sql_query.strip().replace("```sql", "").replace("```", "").strip(";")
            
            if not re.match(r"^\s*SELECT", sql_query, re.IGNORECASE):
                logger.error(f"Generated query is not a SELECT statement: {sql_query}")
                raise ValueError("Generated query is not a SELECT statement.")

            logger.info(f"Generated SQL: {sql_query}")

            # 3. 执行 SQL
            with self.engine.connect() as connection:
                result = connection.execute(text(sql_query))
                rows = [dict(row._mapping) for row in result.fetchall()]
            
            logger.info(f"SQL query returned {len(rows)} rows.")
            
            return {
                "query": query,
                "generated_sql": sql_query,
                "results": rows,
                "error": None
            }

        except Exception as e:
            logger.error(f"Error in SQL Agent execution: {e}")
            return {
                "query": query,
                "generated_sql": sql_query if 'sql_query' in locals() else None,
                "results": [],
                "error": str(e)
            }


    def _initialize_sql_agent(self) -> Callable:
        """
        OPTIMIZED: Initializes the text-to-SQL agent.
        This replaces the mock agent and the placeholder.
        Instead of relying on LangChain's create_sql_agent (which requires
        a compatible LLM object), we use our own LLM client and prompt.
        """
        logger.info("Initializing custom Text-to-SQL agent.")
        # self._run_sql_agent 是我们上面定义的异步方法
        return self._run_sql_agent

    def _fallback_search(self, query: str) -> List[Dict[str, Any]]:
        """
        Fallback search using a simple ILIKE search across tables.
        This is kept as a fallback if the agent fails spectacularly.
        """
        logger.warning(f"Using fallback ILIKE search for query: {query}")
        results = []
        try:
            inspector = inspect(self.engine)
            with self.engine.connect() as connection:
                for table in inspector.get_table_names():
                    for col in inspector.get_columns(table):
                        # 仅在文本类型列上搜索
                        if "VARCHAR" in str(col["type"]) or "TEXT" in str(col["type"]):
                            sql = text(
                                f'SELECT * FROM "{table}" WHERE "{col["name"]}" ILIKE :query LIMIT 5'
                            )
                            res = connection.execute(
                                sql, {"query": f"%{query}%"}
                            ).fetchall()
                            if res:
                                results.extend([dict(row._mapping) for row in res])
            return results
        except SQLAlchemyError as e:
            logger.error(f"Error during fallback ILIKE search: {e}")
            return []

    async def query(self, query: str) -> Dict[str, Any]:
        """
        Primary method to query the tabular database.
        It uses the SQL agent by default.
        """
        if not self.sql_agent:
            logger.error("SQL agent is not initialized. Cannot process query.")
            return {"error": "SQL agent not initialized."}

        try:
            # 异步调用我们的 SQL agent
            result = await self.sql_agent(query)
            
            # 如果 agent 返回错误, 我们可以选择性地触发回退
            if result.get("error"):
                logger.warning(f"SQL agent failed: {result['error']}. Considering fallback.")
                # 可以在这里添加触发 fallback 的逻辑,
                # 但目前我们只返回 agent 的错误
            
            return result

        except Exception as e:
            logger.error(f"Unhandled error during tabular query: {e}")
            return {
                "query": query,
                "results": [],
                "error": f"Unhandled exception: {e}"
            }

    def close(self):
        """Closes the database engine connection pool."""
        if self.engine:
            self.engine.dispose()
            logger.info("Tabular DB client connections closed.")

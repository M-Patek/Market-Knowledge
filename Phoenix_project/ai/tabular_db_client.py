"""
Tabular DB Client for Phoenix.

This module provides a client for interacting with tabular (SQL) databases,
including a text-to-SQL agent capability.
"""

import logging
import re
import asyncio
from typing import Any, List, Dict, Optional, Callable

import sqlalchemy  # type: ignore
from sqlalchemy import create_engine, inspect, text, Engine  # type: ignore
from sqlalchemy.exc import SQLAlchemyError  # type: ignore

from ..api.gemini_pool_manager import GeminiClient
# [Task 4] 导入 PromptManager 和 PromptRenderer
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
        llm_client: GeminiClient, 
        config: Dict[str, Any],
        # [Task 4] 假设这些是由 Registry.py 注入的
        prompt_manager: PromptManager,
        prompt_renderer: PromptRenderer
    ):
        """
        Initializes the TabularDBClient.

        Args:
            db_uri: The SQLAlchemy connection string (e.g., "postgresql://user:pass@host/db").
            llm_client: An instance of the GeminiClient.
            config: Configuration dictionary.
            prompt_manager: [Task 4] Injected PromptManager.
            prompt_renderer: [Task 4] Injected PromptRenderer.
        """
        self.db_uri = db_uri
        self.llm_client = llm_client
        self.config = config.get("tabular_db", {})
        self.engine: Engine = self._create_db_engine()
        self.schema: str = self._get_db_schema()

        # [Task 4] 存储注入的依赖
        self.prompt_manager = prompt_manager
        self.prompt_renderer = prompt_renderer

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
        """[Task 4] 重构：使用 PromptRenderer 生成 SQL 提示。"""
        # 优化：[Task 4] 这个模板现在从 PromptManager 加载
        logger.debug("Rendering text_to_sql prompt...")
        
        try:
            context = {
                "dialect": self.engine.dialect.name,
                "schema": self.schema,
                "query": query
            }
            
            # 1. 渲染整个模板结构
            # (PromptRenderer.render 会调用 prompt_manager.get_prompt)
            # [Task 4] get_prompt 应该返回原始 dict
            rendered_data = self.prompt_renderer.render("text_to_sql", context)
            
            # 2. 提取最终的提示字符串
            # (基于我们的 text_to_sql.json 结构)
            prompt_str = rendered_data.get("full_prompt_template")
            if not prompt_str:
                 raise ValueError("'full_prompt_template' key not found in rendered prompt.")
            
            return prompt_str
            
        except Exception as e:
            logger.error(f"Failed to render text_to_sql prompt: {e}. Using fallback.", exc_info=True)
            # [Task 4] 回退到旧的硬编码逻辑
            dialect_name = self.engine.dialect.name
            return f"""
            You are an expert {dialect_name} SQL query generator.
            Your task is to convert a natural language question into a SQL query
            based on the provided database schema.

            Database Schema:
            {self.schema}

            Rules:
            1. Only generate the SQL query. No preamble, no explanation.
            2. The query must be syntactically correct for {dialect_name}.
            3. Only query tables and columns present in the schema.
            4. Be careful with data types (e.g., casting).
            5. If the question is complex, break it down (e.g., using WITH clauses).
            6. The query should be read-only (SELECT statements only).
            7. Do NOT generate any INSERT, UPDATE, DELETE, or DROP statements.

            Question:
            "{query}"

            SQL Query:
            """

    @retry_with_exponential_backoff(exceptions_to_retry=(SQLAlchemyError,))
    async def _run_sql_agent(self, query: str) -> Dict[str, Any]:
        """
        OPTIMIZED: This is the actual agent execution logic.
        It generates SQL from text, executes it, and returns the result.
        """
        logger.info(f"SQL Agent processing query: {query}")
        sql_query_generated = None
        try:
            # 1. 生成 SQL
            prompt = self._generate_sql_prompt(query)
            # 假设 llm_client.generate_text 是一个异步方法
            sql_query_generated = await self.llm_client.generate_text(prompt)

            if not sql_query_generated:
                raise ValueError("LLM failed to generate SQL query.")

            # 2. 清理和验证 SQL
            sql_query = sql_query_generated.strip().replace("```sql", "").replace("```", "").strip(";")
            
            if not re.match(r"^\s*SELECT", sql_query, re.IGNORECASE):
                logger.error(f"Generated query is not a SELECT statement: {sql_query}")
                raise ValueError("Generated query is not a SELECT statement.")

            logger.info(f"Generated SQL: {sql_query}")

            # 3. 执行 SQL (I/O operation, needs to be retried)
            # Since this is sync I/O inside an async def, we wrap the I/O
            # part in a sync function and run it in a thread.
            def _execute_sync_query():
                with self.engine.connect() as connection:
                    result = connection.execute(text(sql_query))
                    return [dict(row._mapping) for row in result.fetchall()]

            # We apply the retry logic to this async thread execution
            rows = await asyncio.to_thread(_execute_sync_query)
            
            logger.info(f"SQL query returned {len(rows)} rows.")
            
            return {
                "query": query,
                "generated_sql": sql_query,
                "results": rows,
                "error": None
            }

        except SQLAlchemyError as e:
            logger.error(f"SQLAlchemyError in SQL Agent execution: {e}", exc_info=True)
            # This will be caught by the retry decorator
            raise e # Re-raise for the decorator
        except Exception as e:
            logger.error(f"Unexpected error in SQL Agent execution: {e}", exc_info=True)
            # This will NOT be retried by default, which is correct
            return {
                "query": query,
                "generated_sql": sql_query_generated,
                "results": [],
                "error": str(e)
            }


    def _initialize_sql_agent(self) -> Callable:
        """
        OPTIMIZED: Initializes the text-to-SQL agent.
        """
        logger.info("Initializing custom Text-to-SQL agent.")
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
    
    @retry_with_exponential_backoff(exceptions_to_retry=(SQLAlchemyError,))
    async def execute_sql(self, sql: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        [Security Fix] Executes a raw SQL query with parameter binding.
        Safe alternative to running the agent for known structured queries.
        """
        if not self.engine:
             raise ValueError("DB engine not initialized.")
        
        def _exec():
            with self.engine.connect() as conn:
                # SQLAlchemy text() handles parameter binding safely
                result = conn.execute(text(sql), params or {})
                return [dict(row._mapping) for row in result.fetchall()]
        
        return await asyncio.to_thread(_exec)

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
            
            if result.get("error"):
                logger.warning(f"SQL agent failed: {result['error']}. Considering fallback.")
            
            return result

        except Exception as e:
            logger.error(f"Unhandled error during tabular query: {e}")
            return {
                "query": query,
                "results": [],
                "error": f"Unhandled exception: {e}"
            }

    @retry_with_exponential_backoff(exceptions_to_retry=(SQLAlchemyError,))
    async def upsert_data(self, table_name: str, data: Dict[str, Any], unique_key: str) -> bool:
        """
        [Task 2] 插入或更新 (Upsert) 一行数据到指定的表格。
        
        Args:
            table_name (str): 目标表名 (例如 'fundamentals')。
            data (Dict[str, Any]): 要插入的数据 (列名 -> 值)。
            unique_key (str): 用于冲突判断的主键 (例如 'symbol')。
        """
        if not self.engine:
            logger.error(f"Upsert failed: DB engine not initialized.")
            return False
        
        if not data or not unique_key or not table_name:
            logger.error("Upsert failed: table_name, data, and unique_key are required.")
            return False
        
        # (这特定于 PostgreSQL。MySQL/SQLite 使用 'REPLACE' 或 'ON CONFLICT DO UPDATE')
        if self.engine.dialect.name != "postgresql":
            logger.error(f"Upsert logic is only implemented for PostgreSQL, not {self.engine.dialect.name}.")
            return False
        
        # 1. 准备列名和占位符
        columns = data.keys()
        column_names = ", ".join([f'"{c}"' for c in columns])
        placeholders = ", ".join([f":{c}" for c in columns])
        
        # 2. 准备 ON CONFLICT 更新部分
        update_set_parts = [f'"{c}" = EXCLUDED."{c}"' for c in columns if c != unique_key]
        
        if not update_set_parts: # (如果只有 unique_key)
            update_set = "DO NOTHING"
        else:
            update_set = f"DO UPDATE SET {', '.join(update_set_parts)}"
            
        # 3. 构建查询
        sql = text(f"""
            INSERT INTO "{table_name}" ({column_names})
            VALUES ({placeholders})
            ON CONFLICT ("{unique_key}")
            {update_set};
        """)
        
        try:
            # Upsert 是一个 I/O 操作，在异步方法中应在线程中运行
            def _execute_sync():
                with self.engine.connect() as connection:
                    connection.execute(sql, data)
                    connection.commit()
            
            await asyncio.to_thread(_execute_sync)
            
            logger.info(f"Successfully upserted data into '{table_name}' for key {data.get(unique_key)}")
            return True
        except SQLAlchemyError as e:
            logger.error(f"Failed to upsert data into {table_name}: {e}", exc_info=True)
            raise e # Re-raise for the decorator to handle

    def close(self):
        """Closes the database engine connection pool."""
        if self.engine:
            self.engine.dispose()
            logger.info("Tabular DB client connections closed.")

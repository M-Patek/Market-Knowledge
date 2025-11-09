from typing import List, Dict, Any, Optional
import duckdb
import os # 修复：导入 os
import asyncio # 修复：导入 asyncio
# 修复：将 'monitor.logging' 转换为 'Phoenix_project.monitor.logging'
from Phoenix_project.monitor.logging import get_logger

log = get_logger("TabularDBClient")


class TabularDBClient:
    """
    用于与结构化表格数据库（如 DuckDB）交互的客户端。
    """

    # 修复：签名与 ai/tabular_db_client.py (新版) 不匹配
    # def __init__(self, db_path: str, config: Dict[str, Any]):
    def __init__(self, config: Dict[str, Any]):
        
        # 修复：从 config (system.yaml) 获取 db_path
        self.config = config
        self.db_path = self.config.get("db_path", ":memory:") # 默认为内存
        
        # 修复：DuckDB 连接在异步方法中可能存在问题
        # 我们将在每个 async 方法中按需创建连接
        self._db_conn = None 
        # self.db_conn = duckdb.connect(database=self.db_path, read_only=True)
        
        self.tables = [] # 将在 connect_async 中填充
        
        # [✅ 优化] 从配置中获取是否使用 SQL 代理
        self.use_sql_agent = self.config.get("use_sql_agent", False)
        self.sql_agent = None
        
        if self.use_sql_agent:
            # 修复：初始化是同步的，可以在 __init__ 中调用
            self._initialize_sql_agent()

    async def _get_connection(self):
        """ 异步获取或创建 DuckDB 连接 """
        # 警告：DuckDB 的 Python API 本质上是同步的。
        # 在 async 方法中按需创建连接是更安全的方式。
        try:
            # 尝试连接
            # read_only=True 假设 worker/API 是只读的
            conn = await asyncio.to_thread(duckdb.connect, database=self.db_path, read_only=True)
            return conn
        except Exception as e:
            log.error(f"Failed to connect to DuckDB at {self.db_path}: {e}")
            return None

    async def _discover_tables(self) -> List[str]:
        """发现数据库中的所有表。"""
        conn = await self._get_connection()
        if not conn:
            return []
            
        try:
            # 修复：在线程中运行同步的 duckdb 调用
            tables = await asyncio.to_thread(conn.execute("SHOW TABLES").fetchall)
            table_names = [table[0] for table in tables]
            log.info(f"Discovered tables: {table_names}")
            self.tables = table_names # 缓存
            return table_names
        except Exception as e:
            log.error(f"Failed to discover tables: {e}")
            return []
        finally:
            if conn:
                await asyncio.to_thread(conn.close)

    def _initialize_sql_agent(self):
        """
        [✅ 优化] 初始化
        Query-to-SQL 代理。
        这需要额外的依赖 (如 LangChain, Ollama, OpenAI)。
        """
        log.info("Attempting to initialize Query-to-SQL agent...")
        try:
            # 
            # 示例: (这需要 langchain 和一个 LLM)
            # from langchain_community.agent_toolkits import create_sql_agent
            # from langchain_community.utilities import SQLDatabase
            # from langchain_openai import ChatOpenAI
            # 
            # db = SQLDatabase(self.db_conn) # 注意: LangChain 的 SQLDatabase 可能需要 SQLAlchemy URL
            # llm = ChatOpenAI(model="gpt-4", temperature=0)
            # self.sql_agent = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)
            # log.info("Query-to-SQL agent initialized successfully.")
            
            # 
            log.warning("Query-to-SQL agent initialization is a placeholder. "
                        "Full implementation requires LLM and LangChain setup.")
            # 
            self.sql_agent = self._mock_sql_agent # 
            
        except ImportError:
            log.warning("LangChain or LLM dependencies not found. SQL agent disabled.")
            self.use_sql_agent = False
        except Exception as e:
            log.error(f"Failed to initialize SQL agent: {e}")
            self.use_sql_agent = False
            
    # 修复：这是一个同步函数 (LLM 调用可能是异步的，但 mock 是同步的)
    # async def _mock_sql_agent(self, query: str, symbol: str) -> str:
    def _mock_sql_agent(self, query: str, symbol: str) -> str:
        """
        [✅ 优化] 模拟 SQL 代理的行为，用于演示。
        在实际应用中，这将是一个 LLM 调用。
        """
        log.debug(f"Mock SQL Agent processing query: '{query}' for symbol '{symbol}'")
        # 
        # 
        # 
        
        # 
        # 
        if "revenue" in query.lower() and "quarterly" in query.lower():
            sql = f"SELECT period, revenue FROM financials_quarterly WHERE symbol = '{symbol}' ORDER BY period DESC LIMIT 5"
        elif "net income" in query.lower():
            sql = f"SELECT period, net_income FROM financials_annual WHERE symbol = '{symbol}' ORDER BY period DESC LIMIT 3"
        else:
            # 
            log.warning(f"Mock SQL Agent could not generate specific SQL for query: '{query}'. Returning None.")
            return None
        
        log.info(f"Mock SQL Agent generated SQL: {sql}")
        return sql

    async def _fallback_search(self, query: str, symbol: str) -> List[Dict[str, Any]]:
        """
        [✅ 优化] 
        """
        log.debug(f"Using fallback ILIKE search for query: '{query}' on symbol '{symbol}'")
        results = []
        
        conn = await self._get_connection()
        if not conn:
            return []
            
        if not self.tables:
             await self._discover_tables() # 尝试填充

        if not self.tables:
            log.warning("No tables found in database for fallback search.")
            await asyncio.to_thread(conn.close)
            return []

        # 
        search_table = "financials_quarterly" # 
        if search_table not in self.tables:
            # 
            search_table = self.tables[0]
            log.debug(f"'financials_quarterly' not found, falling back to first table: '{search_table}'")

        try:
            # 
            # 
            columns_query = f"PRAGMA table_info('{search_table}')"
            columns_info = await asyncio.to_thread(conn.execute(columns_query).fetchall)
            
            # 
            # 
            text_columns = [col[1] for col in columns_info if "VARCHAR" in col[2].upper()]
            
            description = None # 用于存储列名
            
            if not text_columns:
                log.warning(f"No VARCHAR columns found in table '{search_table}' for ILIKE search.")
                # 
                # 
                all_columns = [col[1] for col in columns_info]
                if not all_columns:
                    await asyncio.to_thread(conn.close)
                    return [] # 
                
                # 
                # 
                query_sql = f"SELECT * FROM {search_table} WHERE symbol = ? LIMIT 10"
                
                # 修复：在线程中运行
                res_obj = await asyncio.to_thread(conn.execute, query_sql, [symbol])
                res = await asyncio.to_thread(res_obj.fetchall)
                description = res_obj.description
            
            else:
                # 
                where_clause = " OR ".join([f"{col} ILIKE ?" for col in text_columns])
                # 
                query_sql = f"SELECT * FROM {search_table} WHERE ({where_clause}) AND symbol = ?"
                
                # 
                like_query = f"%{query}%"
                params = [like_query] * len(text_columns) + [symbol]
                
                # 修复：在线程中运行
                res_obj = await asyncio.to_thread(conn.execute, query_sql, params)
                res = await asyncio.to_thread(res_obj.fetchall)
                description = res_obj.description
            
            # 
            if res:
                column_names = [col[0] for col in description]
                results = [dict(zip(column_names, row)) for row in res]
                
        except Exception as e:
            log.error(f"Fallback search failed: {e}")
        finally:
            if conn:
                await asyncio.to_thread(conn.close)
            
        return results

    async def search_financials(self, query: str, symbol: str) -> List[Dict[str, Any]]:
        """
        使用自然语言查询搜索表格财务数据。
        [✅ 优化] 优先使用 SQL 代理，失败或未配置时回退到 ILIKE 搜索。
        """
        results = []
        sql_query = None
        conn = None # 修复：我们需要一个连接

        try:
            if self.use_sql_agent and self.sql_agent:
                log.debug("Attempting search using SQL Agent.")
                
                # 修复：SQL Agent (mock) 是同步的
                sql_query = await asyncio.to_thread(self.sql_agent, query, symbol)
                    
                if sql_query:
                    # 
                    conn = await self._get_connection()
                    if not conn:
                         raise ConnectionError("Failed to get DB connection for SQL Agent query")
                         
                    # 修复：在线程中运行
                    res_obj = await asyncio.to_thread(conn.execute, sql_query)
                    res = await asyncio.to_thread(res_obj.fetchall)
                    
                    if res:
                        column_names = [col[0] for col in res_obj.description]
                        results = [dict(zip(column_names, row)) for row in res]
                else:
                    # 
                    log.warning("SQL Agent returned no query, falling back.")
                    results = await self._fallback_search(query, symbol)

            else:
                log.debug("SQL Agent not enabled. Using fallback ILIKE search.")
                results = await self._fallback_search(query, symbol)
                
        except Exception as e:
            log.error(f"SQL Agent search failed: {e}. Falling back to ILIKE search.")
            results = await self._fallback_search(query, symbol)
        finally:
             if conn:
                await asyncio.to_thread(conn.close)

        return results

    async def query(self, sql_query: str) -> List[Dict[str, Any]]:
        """
        执行一个原始 SQL 查询。
        """
        conn = await self._get_connection()
        if not conn:
            return []
            
        try:
            # 修复：在线程中运行
            res_obj = await asyncio.to_thread(conn.execute, sql_query)
            res = await asyncio.to_thread(res_obj.fetchall)
            
            if res:
                column_names = [col[0] for col in res_obj.description]
                return [dict(zip(column_names, row)) for row in res]
            return []
        except Exception as e:
            log.error(f"Raw SQL query failed: {e}")
            return []
        finally:
            if conn:
                await asyncio.to_thread(conn.close)

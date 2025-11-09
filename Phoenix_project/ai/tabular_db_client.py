from typing import List, Dict, Any, Optional
import duckdb
from phoenix_project.monitor.logging import get_logger

log = get_logger("TabularDBClient")


class TabularDBClient:
    """
    用于与结构化表格数据库（如 DuckDB）交互的客户端。
    """

    def __init__(self, db_path: str, config: Dict[str, Any]):
        self.db_path = db_path
        self.config = config
        self.db_conn = duckdb.connect(database=self.db_path, read_only=True)
        self.tables = self._discover_tables()
        
        # [✅ 优化] 从配置中获取是否使用 SQL 代理
        self.use_sql_agent = self.config.get("use_sql_agent", False)
        self.sql_agent = None
        
        if self.use_sql_agent:
            self._initialize_sql_agent()

    def _discover_tables(self) -> List[str]:
        """发现数据库中的所有表。"""
        try:
            tables = self.db_conn.execute("SHOW TABLES").fetchall()
            table_names = [table[0] for table in tables]
            log.info(f"Discovered tables: {table_names}")
            return table_names
        except Exception as e:
            log.error(f"Failed to discover tables: {e}")
            return []

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

    def _fallback_search(self, query: str, symbol: str) -> List[Dict[str, Any]]:
        """
        [✅ 优化] 
        """
        log.debug(f"Using fallback ILIKE search for query: '{query}' on symbol '{symbol}'")
        results = []
        
        # 
        if not self.tables:
            log.warning("No tables found in database for fallback search.")
            return []

        # 
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
            columns_info = self.db_conn.execute(columns_query).fetchall()
            
            # 
            # 
            text_columns = [col[1] for col in columns_info if "VARCHAR" in col[2].upper()]
            
            if not text_columns:
                log.warning(f"No VARCHAR columns found in table '{search_table}' for ILIKE search.")
                # 
                # 
                all_columns = [col[1] for col in columns_info]
                if not all_columns:
                    return [] # 
                
                # 
                # 
                query_sql = f"SELECT * FROM {search_table} WHERE symbol = ? LIMIT 10"
                res = self.db_conn.execute(query_sql, [symbol]).fetch_all()
            
            else:
                # 
                where_clause = " OR ".join([f"{col} ILIKE ?" for col in text_columns])
                # 
                query_sql = f"SELECT * FROM {search_table} WHERE ({where_clause}) AND symbol = ?"
                
                # 
                like_query = f"%{query}%"
                params = [like_query] * len(text_columns) + [symbol]
                
                res = self.db_conn.execute(query_sql, params).fetch_all()
            
            # 
            if res:
                column_names = [col[0] for col in self.db_conn.description]
                results = [dict(zip(column_names, row)) for row in res]
                
        except Exception as e:
            log.error(f"Fallback search failed: {e}")
            
        return results

    def search_financials(self, query: str, symbol: str) -> List[Dict[str, Any]]:
        """
        使用自然语言查询搜索表格财务数据。
        [✅ 优化] 优先使用 SQL 代理，失败或未配置时回退到 ILIKE 搜索。
        """
        results = []
        sql_query = None

        if self.use_sql_agent and self.sql_agent:
            log.debug("Attempting search using SQL Agent.")
            try:
                # 
                sql_query = self.sql_agent(query, symbol)
                
                if sql_query:
                    # 
                    res = self.db_conn.execute(sql_query).fetch_all()
                    if res:
                        column_names = [col[0] for col in self.db_conn.description]
                        results = [dict(zip(column_names, row)) for row in res]
                else:
                    # 
                    log.warning("SQL Agent returned no query, falling back.")
                    results = self._fallback_search(query, symbol)

            except Exception as e:
                log.error(f"SQL Agent search failed: {e}. Falling back to ILIKE search.")
                results = self._fallback_search(query, symbol)
        else:
            log.debug("SQL Agent not enabled. Using fallback ILIKE search.")
            results = self._fallback_search(query, symbol)

        return results

    def query(self, sql_query: str) -> List[Dict[str, Any]]:
        """
        执行一个原始 SQL 查询。
        """
        try:
            res = self.db_conn.execute(sql_query).fetch_all()
            if res:
                column_names = [col[0] for col in self.db_conn.description]
                return [dict(zip(column_names, row)) for row in res]
            return []
        except Exception as e:
            log.error(f"Raw SQL query failed: {e}")
            return []

import asyncpg
from typing import Dict, Any, Optional, List, Tuple
# 修复：将相对导入 'from ..monitor.logging...' 更改为绝对导入
from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

class TabularDBClient:
    """
    Client for interacting with a tabular database (e.g., PostgreSQL).
    Used for storing and retrieving structured financial data (e.g., fundamentals).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the TabularDBClient.
        
        Args:
            config (Dict[str, Any]): Configuration dict, expects 'tabular_db'
                                      with 'dsn' (Database Source Name).
        """
        # [蓝图 2 修复]：配置现在是 config['tabular_db']
        db_config = config # 假设 config 已经是 config['tabular_db']
        self.dsn = db_config.get('dsn') # e.g., "postgresql://user:pass@host:port/db"
        if not self.dsn:
            logger.error("TabularDB 'dsn' not configured.")
            raise ValueError("TabularDB 'dsn' is required.")
            
        self.pool = None
        logger.info("TabularDBClient initialized.")

    async def connect(self):
        """Creates the connection pool."""
        if self.pool:
            return
        try:
            self.pool = await asyncpg.create_pool(dsn=self.dsn, min_size=2, max_size=10)
            logger.info("TabularDB connection pool created.")
        except Exception as e:
            logger.error(f"Failed to create TabularDB connection pool: {e}", exc_info=True)
            raise

    async def close(self):
        """Closes the connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None
            logger.info("TabularDB connection pool closed.")

    # --- [任务 2 实现] ---
    # 移除了模拟数据，实现了真实的 asyncpg 查询
    async def query_metric(self, symbol: str, metric_name: str) -> Optional[Any]:
        """
        Fetches the latest value for a specific metric and symbol.
        This implements the query capability described in RAG_ARCHITECTURE.md.
        
        Args:
            symbol (str): The ticker symbol (e.g., 'AAPL').
            metric_name (str): The metric name (e.g., 'Revenue', 'eps').
            
        Returns:
            Optional[Any]: The value of the metric, or None if not found.
        """
        if not self.pool:
            logger.error("Connection pool is not initialized. Call connect() first.")
            await self.connect() # 尝试重新连接
            if not self.pool:
                 return None
            
        # 此查询基于 RAG_ARCHITECTURE.md 中的设计
        # 它假设有一个包含 [symbol, metric_name, metric_value, report_date] 的表
        query = """
        SELECT metric_value FROM financial_metrics
        WHERE symbol = $1 AND metric_name = $2
        ORDER BY report_date DESC
        LIMIT 1;
        """
        
        try:
            async with self.pool.acquire() as connection:
                # fetchval() 直接获取第一行第一列的值
                value = await connection.fetchval(query, symbol, metric_name)
            
            if value is not None:
                logger.debug(f"Successfully fetched metric {metric_name} for {symbol}: {value}")
                return value
            else:
                logger.debug(f"No metric '{metric_name}' found for symbol: {symbol}")
                return None
        except asyncpg.exceptions.UndefinedTableError:
            logger.error("Query failed: 'financial_metrics' table does not exist.")
            return None
        except Exception as e:
            logger.error(f"Error querying TabularDB for {symbol} metric {metric_name}: {e}", exc_info=True)
            return None
    # --- [任务 2 结束] ---

    async def get_latest_financials(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Example query: Fetches the latest financial metrics for a given symbol.
        
        This assumes a table named 'financial_metrics' exists.
        
        Args:
            symbol (str): The ticker symbol.
            
        Returns:
            Optional[Dict[str, Any]]: A dictionary of metrics, or None.
        """
        if not self.pool:
            logger.error("Connection pool is not initialized. Call connect() first.")
            await self.connect() # 尝试重新连接
            if not self.pool:
                 return None
            
        query = """
        SELECT * FROM financial_metrics
        WHERE symbol = $1
        ORDER BY report_date DESC
        LIMIT 1;
        """
        
        try:
            async with self.pool.acquire() as connection:
                record = await connection.fetchrow(query, symbol)
            
            if record:
                # Convert asyncpg.Record to a plain dict
                return dict(record)
            else:
                logger.debug(f"No financial metrics found for symbol: {symbol}")
                return None
        except asyncpg.exceptions.UndefinedTableError:
            logger.error("Query failed: 'financial_metrics' table does not exist.")
            return None
        except Exception as e:
            logger.error(f"Error querying TabularDB for {symbol}: {e}", exc_info=True)
            return None

    # --- [蓝图 2] 新增 RAG 搜索方法 ---
    async def search_financials(
        self, 
        query: str, 
        symbols: List[str], 
        limit: int = 10
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Searches financial metrics relevant to a query and symbols.
        For RRF, we need a list of (doc, score).
        We will return matching rows and assign a static score (e.g., 1.0)
        as relevance is binary (it's a structured fact).
        
        Args:
            query (str): The search query (used to find matching metric_names).
            symbols (List[str]): List of ticker symbols.
            limit (int): Max number of rows to return.
            
        Returns:
            List[Tuple[Dict[str, Any], float]]: List of (row_dict, score) tuples.
        """
        if not self.pool:
            logger.error("Connection pool is not initialized. Call connect() first.")
            await self.connect() # 尝试重新连接
            if not self.pool:
                 return []
            
        # 这是一个简化的搜索。我们将在查询中搜索 metric_name。
        # 一个更高级的系统会使用 query_to_sql_agent 或 embedding on metric names。
        # 我们还按符号过滤。
        
        # 简化：假设查询是 "revenue" 或 "eps"
        # 我们将在 financial_metrics 中搜索。
        
        # Psycopg (asyncpg) 不支持 'IN %s' 绑定。我们必须使用 ANY($1)
        params = []
        param_idx = 1
        
        symbol_query = ""
        if symbols:
            symbol_query = f"WHERE symbol = ANY(${param_idx})"
            params.append(symbols)
            param_idx += 1
        
        # 动态添加 LIKE 过滤器（如果 query 存在）
        # 这很基础，但可以用于演示
        like_query = ""
        if query:
            # 假设查询是单个词
            query_param = f"%{query.lower().split()[0]}%"
            like_query = f"AND metric_name ILIKE ${param_idx}"
            params.append(query_param)
            param_idx += 1
            
            # 如果没有符号，WHERE 子句需要调整
            if not symbols:
                 like_query = like_query.replace("AND", "WHERE")


        # [蓝图 2] 假设的查询
        sql_query = f"""
        SELECT * FROM financial_metrics
        {symbol_query}
        {like_query}
        ORDER BY report_date DESC
        LIMIT {limit};
        """
        
        try:
            async with self.pool.acquire() as connection:
                records = await connection.fetch(sql_query, *params)
            
            results = []
            for record in records:
                # 对于 RRF，我们返回 (文档, 分数)
                # 表格事实是 100% 相关的
                results.append((dict(record), 1.0))
                
            return results
        except asyncpg.exceptions.UndefinedTableError:
            logger.error("Query failed: 'financial_metrics' table does not exist.")
            return []
        except Exception as e:
            logger.error(f"Error searching financials for {symbols}: {e}", exc_info=True)
            return []
    # --- [蓝图 2 结束] ---

    async def upsert_financials(self, symbol: str, metrics: Dict[str, Any]) -> bool:
        """
        Example upsert: Inserts or updates financial metrics for a symbol.
        This is highly specific to the table schema.
        
        Args:
            symbol (str): The ticker symbol.
            metrics (Dict[str, Any]): Dict of metrics (e.g., {"eps": 1.25, "pe_ratio": 20.5, ...})
            
        Returns:
            bool: True on success, False on failure.
        """
        if not self.pool:
            logger.error("Connection pool is not initialized. Call connect() first.")
            await self.connect() # 尝试重新连接
            if not self.pool:
                 return False

        # This is a simplified example. A real query would be more robust.
        # It assumes 'symbol' and 'report_date' is a composite key.
        # This query is also vulnerable to schema mismatch.
        
        # TODO: Dynamically build columns and values
        # For this example, we'll assume a few fixed columns
        cols_to_insert = ['symbol', 'report_date', 'eps', 'pe_ratio', 'metric_name', 'metric_value'] # 扩展
        
        # Ensure 'symbol' is in the dict
        metrics['symbol'] = symbol
        
        # Prepare data, using None for missing keys
        values = [metrics.get(col) for col in cols_to_insert]
        
        # e.g., ($1, $2, $3, $4, $5, $6)
        placeholders = ", ".join([f"${i+1}" for i in range(len(cols_to_insert))])
        
        # e.g., eps = EXCLUDED.eps, pe_ratio = EXCLUDED.pe_ratio ...
        update_set = ", ".join([f"{col} = EXCLUDED.{col}" for col in cols_to_insert if col not in ['symbol', 'report_date']]) # 假设 (symbol, report_date) 是键

        query = f"""
        INSERT INTO financial_metrics ({", ".join(cols_to_insert)})
        VALUES ({placeholders})
        ON CONFLICT (symbol, report_date) DO UPDATE -- Assumes composite key
        SET {update_set};
        """
        
        try:
            async with self.pool.acquire() as connection:
                await connection.execute(query, *values)
            return True
        except asyncpg.exceptions.UndefinedTableError:
            logger.error("Query failed: 'financial_metrics' table does not exist.")
            return False
        except Exception as e:
            logger.error(f"Error upserting financials for {symbol}: {e}", exc_info=True)
            return False

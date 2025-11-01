import asyncpg
from typing import Dict, Any, Optional
from ..monitor.logging import get_logger

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
        db_config = config.get('tabular_db', {})
        self.dsn = db_config.get('dsn') # e.g., "postgresql://user:pass@host:port/db"
        if not self.dsn:
            logger.error("TabularDB 'dsn' not configured.")
            raise ValueError("TabularDB 'dsn' is required.")
            
        self.pool = None
        logger.info("TabularDBClient initialized.")

    async def connect(self):
        """Creates the connection pool."""
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
            logger.info("TabularDB connection pool closed.")

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
            return False

        # This is a simplified example. A real query would be more robust.
        # It assumes 'symbol' is a primary key or has a unique constraint.
        # This query is also vulnerable to schema mismatch.
        
        # TODO: Dynamically build columns and values
        # For this example, we'll assume a few fixed columns
        cols_to_insert = ['symbol', 'report_date', 'eps', 'pe_ratio']
        
        # Ensure 'symbol' is in the dict
        metrics['symbol'] = symbol
        
        # Prepare data, using None for missing keys
        values = [metrics.get(col) for col in cols_to_insert]
        
        # e.g., ($1, $2, $3, $4)
        placeholders = ", ".join([f"${i+1}" for i in range(len(cols_to_insert))])
        
        # e.g., eps = EXCLUDED.eps, pe_ratio = EXCLUDED.pe_ratio
        update_set = ", ".join([f"{col} = EXCLUDED.{col}" for col in cols_to_insert if col != 'symbol'])

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
        except Exception as e:
            logger.error(f"Error upserting financials for {symbol}: {e}", exc_info=True)
            return False

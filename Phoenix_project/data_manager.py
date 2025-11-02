from typing import Dict, Any, List, Optional
import pandas as pd
from core.schemas.data_schema import MarketData, NewsData, EconomicIndicator
from config.loader import ConfigLoader
from data.data_iterator import DataIterator
from monitor.logging import get_logger

logger = get_logger(__name__)

class DataManager:
    """
    Handles loading, caching, and serving all data required by the system.
    It abstracts the data sources (e.g., APIs, databases, local files).
    """

    def __init__(self, config_loader: ConfigLoader):
        self.config_loader = config_loader
        self.data_catalog = self._load_data_catalog()
        
        # In-memory cache for loaded data (e.g., for backtesting)
        self.data_cache: Dict[str, pd.DataFrame] = {}
        
        logger.info(f"DataManager initialized. Found {len(self.data_catalog)} entries in catalog.")

    def _load_data_catalog(self) -> Dict[str, Any]:
        """Loads the data catalog file."""
        try:
            # Assuming data_catalog.json is in the root
            catalog = self.config_loader.load_json("data_catalog.json")
            return catalog.get("datasets", {})
        except Exception as e:
            logger.error(f"Failed to load data_catalog.json: {e}", exc_info=True)
            return {}

    def _load_dataset(self, dataset_id: str) -> Optional[pd.DataFrame]:
        """
        Loads a single dataset from the path specified in the catalog.
        Caches the result in memory.
        """
        if dataset_id in self.data_cache:
            return self.data_cache[dataset_id]
            
        if dataset_id not in self.data_catalog:
            logger.error(f"Dataset '{dataset_id}' not found in data catalog.")
            return None
            
        dataset_info = self.data_catalog[dataset_id]
        file_path = dataset_info.get("path")
        file_type = dataset_info.get("type", "csv")
        
        if not file_path:
            logger.error(f"No path specified for dataset '{dataset_id}'.")
            return None
            
        try:
            # TODO: Handle relative paths correctly from project root
            # Assuming file_path is relative to project root
            
            if file_type == "csv":
                df = pd.read_csv(file_path, parse_dates=["timestamp"])
            elif file_type == "parquet":
                df = pd.read_parquet(file_path)
                if "timestamp" in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
            else:
                logger.error(f"Unsupported file type '{file_type}' for dataset '{dataset_id}'.")
                return None
                
            logger.info(f"Successfully loaded dataset '{dataset_id}' from {file_path}. Shape: {df.shape}")
            self.data_cache[dataset_id] = df
            return df
            
        except FileNotFoundError:
            logger.error(f"Data file not found: {file_path}")
            return None
        except Exception as e:
            logger.error(f"Failed to load dataset '{dataset_id}': {e}", exc_info=True)
            return None

    def get_backtest_iterator(self, market_data_ids: List[str], news_data_id: Optional[str] = None) -> DataIterator:
        """
        Pre-loads all necessary data for a backtest and returns
        a DataIterator instance.
        
        Args:
            market_data_ids (List[str]): List of dataset IDs for market data.
                                         The catalog entry *must* contain a 'symbol'.
            news_data_id (Optional[str]): Dataset ID for news data.
            
        Returns:
            DataIterator
        """
        logger.info("Preparing backtest data iterator...")
        
        market_data_dfs: Dict[str, pd.DataFrame] = {}
        for dataset_id in market_data_ids:
            symbol = self.data_catalog.get(dataset_id, {}).get("symbol")
            if not symbol:
                logger.error(f"Dataset '{dataset_id}' in catalog has no 'symbol' defined. Skipping.")
                continue
                
            df = self._load_dataset(dataset_id)
            if df is not None:
                market_data_dfs[symbol] = df
                
        news_data_df = None
        if news_data_id:
            news_data_df = self._load_dataset(news_data_id)
            
        if not market_data_dfs:
            raise ValueError("No valid market data could be loaded for the backtest.")
            
        return DataIterator(
            market_data_dfs=market_data_dfs,
            news_data_df=news_data_df
        )

    async def fetch_live_data(self) -> Dict[str, List[Any]]:
        """
        Fetches new data from live APIs.
        This is a placeholder for a real implementation.
        
        Returns:
            A dictionary of data lists, e.g.:
            {"market_data": [MarketData(...)], "news_data": [NewsData(...)]}
        """
        logger.warning("fetch_live_data is a placeholder and not implemented.")
        # --- Placeholder Implementation ---
        # 1. Connect to live data provider (e.g., Alpaca, Polygon)
        # 2. Get data since `last_data_ingest_time`
        # 3. Parse data into Pydantic models
        # 4. Return the batch
        
        await asyncio.sleep(0.1) # Simulate async I/O
        
        # Example dummy data
        dummy_market_data = MarketData(
            symbol="DUMMY",
            timestamp=datetime.utcnow(),
            open=100, high=101, low=99, close=100.5, volume=10000
        )
        
        return {
            "market_data": [dummy_market_data],
            "news_data": [],
            "economic_data": []
        }

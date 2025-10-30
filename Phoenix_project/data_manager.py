import logging
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from datetime import date
from typing import Dict, Any, List, Optional
from schemas.data_schema import DataSchema

from storage.s3_client import S3Client
from features.store import FeatureStore

class IDataManager(ABC):
    @abstractmethod
    def load_market_data(self, source: str, tickers: List[str], start_date: str, end_date: str) -> List[DataSchema]:
        pass
    
    @abstractmethod
    def get_features_for_date(self, dt: date) -> Dict[str, Any]:
        pass

class DataCache:
    """Manages in-memory caching for frequently accessed data."""
    def __init__(self):
        self.market_data_cache = {}
        self.feature_cache = {}

    def get_market_data(self, key: str) -> Optional[List[DataSchema]]:
        return self.market_data_cache.get(key)

    def set_market_data(self, key: str, data: List[DataSchema]):
        self.market_data_cache[key] = data

    def get_feature(self, key: str) -> Optional[Any]:
        return self.feature_cache.get(key)

    def set_feature(self, key: str, data: Any):
        self.feature_cache[key] = data

class DataManager(IDataManager):
    """
    [V2.0] Refactored DataManager
    - Manages all data loading (market, alternative)
    - Manages feature calculation (via FeatureStore)
    - Manages caching (via DataCache)
    - Ensures all outputs conform to schemas (DataSchema, FeatureSchema)
    """
    def __init__(self, s3_client: S3Client, feature_store: FeatureStore, feature_cache_dir: str):
        self.logger = logging.getLogger("PhoenixProject.DataManager")
        self.s3_client = s3_client
        self.feature_store = feature_store
        self.feature_cache_dir = feature_cache_dir # For persistent cache
        
        # In-memory cache for this session
        self.market_data_cache = {}

        self.logger.info(f"DataManager initialized. Feature cache directory: {self.feature_cache_dir}")

    def load_market_data(self, source: str, tickers: List[str], start_date: str, end_date: str) -> List[DataSchema]:
        """
        (L1 Patched) 从指定来源（例如 S3）加载市场数据。
        Returns List[DataSchema]
        """
        cache_key = f"{source}_{'_'.join(tickers)}_{start_date}_{end_date}"
        if cache_key in self.market_data_cache: # In a real scenario, this should return List[DataSchema] too
            self.logger.debug(f"Loading market data from in-memory cache for key: {cache_key}")
            return self.market_data_cache[cache_key]

        self.logger.debug(f"Fetching market data for key: {cache_key}")
        
        # --- Placeholder: Mock Data Generation ---
        # In a real system:
        # data_df = self.s3_client.load_data(bucket, f"market_data/{source}/{...}.parquet")
        
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        if date_range.empty:
             self.logger.warning(f"No date range generated for {start_date} to {end_date}")
             return []
        data = {
            "timestamp": pd.to_datetime(date_range),
            "price": np.random.rand(len(date_range)) * 100 + 500
        }
        df = pd.DataFrame(data)
        
        if df.empty:
            self.logger.warning(f"No market data found for key: {cache_key}")
            return []

        # (L1) Convert DataFrame to List[DataSchema]
        data_schemas = [
            DataSchema(
                timestamp=row['timestamp'],
                source=source,
                symbol=tickers[0], # Mocking: assuming one ticker for simplicity
                value=row['price']
            ) for _, row in df.iterrows()
        ]

        self.market_data_cache[cache_key] = data_schemas
        return data_schemas

    def get_features_for_date(self, dt: date) -> Dict[str, Any]:
        """
        [Task 2.1] Calculate (or load from cache) all registered features
        for a specific date.
        """
        # 1. Check persistent cache (e.g., /cache/features/2023-10-27.json)
        # ... logic to load from file system ...
        
        # 2. If not in cache, compute them
        self.logger.debug(f"Cache miss. Generating features for date: {dt}")
        
        # 2a. Load necessary raw data for this date
        # (This is a simplified view; we'd need a sliding window of raw data)
        # raw_market_data = self.load_market_data(...)
        # raw_alt_data = self.load_alternative_data(...)
        
        # 2b. Compute features using the store
        # features = self.feature_store.generate_features_for_ticker(...)
        
        # 3. Save to cache
        # ... logic to save to file system ...
        
        # Placeholder: Return mock features
        return {
            "sma_50": 150.0 + np.random.rand(),
            "rsi_14": 45.0 + np.random.rand() * 10,
            "supply_chain_risk": 0.2 + np.random.rand() * 0.1
        }

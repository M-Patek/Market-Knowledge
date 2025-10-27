import os
import logging
from collections import defaultdict
import pandas as pd
from datetime import date
# from storage.s3_client import S3Client
from typing import List

class DataManager:
    def __init__(self, config, providers, cold_storage_client=None):
        self.logger = logging.getLogger("PhoenixProject.DataManager")
        self.config = config
        self.cache_dir = config.get('cache_dir', '/tmp/phoenix_cache')
        self.providers = providers # Dict of {data_type: provider_instance}
        self.cold_storage = cold_storage_client
        self.data_catalog = self._load_data_catalog()
        
        # In-memory cache for hot data (e.g., last N days)
        self.hot_cache = defaultdict(pd.DataFrame)
        
        os.makedirs(self.cache_dir, exist_ok=True)
        self.logger.info(f"DataManager initialized. Cache directory set to {self.cache_dir}")

    def _load_data_catalog(self):
        # In a real system, this would load from a file or DB
        return {
            "SPY_1D": {"source": "provider_A", "asset": "SPY", "frequency": "1D"},
            "QQQ_1H": {"source": "provider_B", "asset": "QQQ", "frequency": "1H"},
        }

    def _get_hot_cache_path(self, data_key: str) -> str:
        """Generates a standardized file path for a hot cache key."""
        return os.path.join(self.cache_dir, f"{data_key}.csv")

    def load_data(self, data_key: str, start_date=None, end_date=None) -> pd.DataFrame:
        """
        Loads data from the hot cache if available, otherwise fetches from the provider.
        """
        self.logger.info(f"Requesting data for '{data_key}' from {start_date} to {end_date}.")
        
        hot_cache_path = self._get_hot_cache_path(data_key)
        
        # 1. Try loading from the local hot cache file
        if os.path.exists(hot_cache_path):
            self.logger.debug(f"Loading '{data_key}' from hot cache file: {hot_cache_path}")
            try:
                cached_data = pd.read_csv(hot_cache_path, index_col=0, parse_dates=True)
                # TODO: Add logic to check if the cached data covers the requested date range
                return cached_data
            except Exception as e:
                self.logger.warning(f"Could not read cache file {hot_cache_path}: {e}. Fetching from provider.")
                
        # 2. If not in cache, fetch from the appropriate provider
        catalog_entry = self.data_catalog.get(data_key)
        if not catalog_entry:
            self.logger.error(f"No entry found in data catalog for key: {data_key}")
            return pd.DataFrame()

        provider_name = catalog_entry.get("source")
        provider = self.providers.get(provider_name)
        
        if not provider:
            self.logger.error(f"No provider instance found for source: {provider_name}")
            return pd.DataFrame()
            
        self.logger.info(f"Fetching '{data_key}' from provider '{provider_name}'...")
        data = provider.fetch_data(
            asset=catalog_entry.get("asset"),
            frequency=catalog_entry.get("frequency"),
            start=start_date,
            end=end_date
        )
        
        # 3. Save the fetched data to the hot cache
        if not data.empty:
            data.to_csv(hot_cache_path)
            self.logger.info(f"Saved fetched data for '{data_key}' to hot cache.")
            
        return data

    def archive_hot_cache(self, data_key: str, archive_before_date):
        """
        Moves old data from the hot cache to cold storage (e.g., S3).
        """
        if not self.cold_storage:
            self.logger.info("No cold storage client configured. Skipping archival.")
            return
            
        hot_cache_path = self._get_hot_cache_path(data_key)
        if not os.path.exists(hot_cache_path):
            self.logger.info(f"No hot cache file for '{data_key}'. Skipping archival.")
            return
            
        full_data = pd.read_csv(hot_cache_path, index_col=0, parse_dates=True)
        
        data_to_archive = full_data[full_data.index < archive_before_date]
        remaining_hot_data = full_data[full_data.index >= archive_before_date]
        
        if not data_to_archive.empty:
            cold_storage_key = f"historical_data/{data_key}/{archive_before_date.year}.parquet"
            self.logger.info(f"Archiving {len(data_to_archive)} rows of '{data_key}' to {cold_storage_key}...")
            # In a real system, we'd append to the parquet file or use partitions
            # Logic to append/update data in S3
            self.cold_storage.write_data(cold_storage_key, data_to_archive)
            # Rewrite the hot cache with only recent data
            remaining_hot_data.to_csv(hot_cache_path)

    def save_ai_features(self, features_dataframe: pd.DataFrame):
        """
        [Sub-Task 2.1.1] Saves a dataframe of AI features to a partitioned Parquet structure.
        """
        if 'timestamp' not in features_dataframe.columns:
            self.logger.error("AI features dataframe must contain a 'timestamp' column for partitioning.")
            return

        # Ensure timestamp is a datetime object
        features_dataframe['timestamp'] = pd.to_datetime(features_dataframe['timestamp'])

        for date_group, group_df in features_dataframe.groupby(pd.Grouper(key='timestamp', freq='D')):
            if group_df.empty:
                continue
            
            date_obj = date_group.date()
            partition_path = os.path.join(self.cache_dir, 'features', str(date_obj.year), f"{date_obj.month:02d}", f"{date_obj.day:02d}")
            os.makedirs(partition_path, exist_ok=True)
            
            file_path = os.path.join(partition_path, 'data.parquet')
            group_df.to_parquet(file_path)
            self.logger.info(f"Saved {len(group_df)} AI features to '{file_path}'.")

    def load_ai_features(self, date_range: List[date], asset_ids: List[str]) -> pd.DataFrame:
        """
        [Sub-Task 2.1.2] Loads AI features from the partitioned Parquet cache for a given date range and assets.
        """
        self.logger.info(f"Loading AI features for {len(asset_ids)} assets from {date_range[0]} to {date_range[-1]}...")
        # TODO: Implement the logic to scan the partitioned directory structure,
        # read the relevant parquet files, filter by asset_ids, and concatenate them into a single DataFrame.
        return pd.DataFrame() # Return empty DataFrame for now

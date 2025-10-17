# Phoenix_project/data_manager.py
import os
from datetime import timedelta, date, datetime
from collections import defaultdict
import pandas as pd
# from storage.s3_client import S3Client

class DataManager:
    def __init__(self, config, providers, cold_storage_client=None):
        self.config = config
        self.providers = sorted(providers, key=lambda p: p.get_config().get('cost_per_request', float('inf')))
        self.cache_dir = config.get('data_cache_dir', 'data_cache')
        self.hot_tier_days = config.get('hot_tier_days', 365)
        self.usage_stats = defaultdict(lambda: {'count': 0, 'date': date.today()})
        self.cold_storage = cold_storage_client # The new S3Client instance
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_available_provider(self):
        today = date.today()
        for provider in self.providers:
            provider_name = provider.get_config()['name']
            stats = self.usage_stats[provider_name]
            
            # Reset daily usage count if a new day has started
            if stats['date'] != today:
                stats['count'] = 0
                stats['date'] = today

            if stats['count'] < provider.get_config().get('daily_limit', float('inf')):
                return provider
        return None # No providers available

    def fetch_historical_data(self, ticker, start_date, end_date):
        """
        Fetches data, intelligently querying across hot (local cache)
        and cold (S3) storage tiers.
        """
        hot_cache_path = f"{self.cache_dir}/{ticker}.csv"
        cold_storage_key = f"historical_data/{ticker}.parquet"

        hot_data = pd.DataFrame()
        cold_data = pd.DataFrame()

        # 1. Try to read from hot cache
        if os.path.exists(hot_cache_path):
            hot_data = pd.read_csv(hot_cache_path, index_col=0, parse_dates=True)

        # 2. If data is still missing, try to read from cold storage
        if self.cold_storage:
            # This logic would be more complex, checking if the date range falls in the cold tier
            cold_data = self.cold_storage.read_data(cold_storage_key)
        
        # 3. Combine data from both tiers
        combined_data = pd.concat([cold_data, hot_data]).drop_duplicates()
        if not combined_data.empty:
            combined_data = combined_data.sort_index()

        # 4. Fetch any remaining missing data using the incremental update logic
        last_date_available = combined_data.index.max().date() if not combined_data.empty else None
        
        if last_date_available and last_date_available >= end_date.date():
             print(f"Data for {ticker} is already up to date from cache/cold storage.")
             return combined_data.loc[start_date:end_date]

        fetch_start_date = last_date_available + timedelta(days=1) if last_date_available else start_date
        
        provider = self._get_available_provider()
        if provider:
            new_data = provider.fetch(ticker, fetch_start_date, end_date)
            provider_config = provider.get_config()
            self.usage_stats[provider_config['name']]['count'] += 1
            if new_data is not None and not new_data.empty:
                 combined_data = pd.concat([combined_data, new_data]).drop_duplicates()
                 # Update the hot cache with the newly fetched data
                 cutoff = datetime.now().date() - timedelta(days=self.hot_tier_days)
                 combined_data[combined_data.index.date >= cutoff].to_csv(hot_cache_path)

        return combined_data.loc[start_date:end_date]

    def archive_cold_data(self, ticker):
        """
        Migrates data older than the 'hot_tier_days' threshold
        from the local cache to cold storage.
        This would be run periodically by a separate maintenance process.
        """
        if not self.cold_storage:
            print("Cold storage client not configured. Archiving skipped.")
            return

        hot_cache_path = f"{self.cache_dir}/{ticker}.csv"
        if not os.path.exists(hot_cache_path):
            return

        hot_data = pd.read_csv(hot_cache_path, index_col=0, parse_dates=True)
        cutoff_date = date.today() - timedelta(days=self.hot_tier_days)

        data_to_archive = hot_data[hot_data.index.date() < cutoff_date]
        remaining_hot_data = hot_data[hot_data.index.date() >= cutoff_date]

        if not data_to_archive.empty:
            print(f"Archiving {len(data_to_archive)} rows of cold data for {ticker} to S3.")
            cold_storage_key = f"historical_data/{ticker}.parquet"
            # Logic to append/update data in S3
            self.cold_storage.write_data(cold_storage_key, data_to_archive)
            # Rewrite the hot cache with only recent data
            remaining_hot_data.to_csv(hot_cache_path)

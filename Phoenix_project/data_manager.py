import pandas as pd
import yfinance as yf
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from .monitor.logging import get_logger

logger = get_logger(__name__)

class DataManager:
    """
    Provides a standardized interface for fetching historical market data.
    
    This implementation uses `yfinance` as the data source.
    It can be (and should be) subclassed to support other sources
    (e.g., AlphaVantage, a local database, S3).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the DataManager.
        
        Args:
            config (Dict, Any): The main system configuration.
        """
        self.config = config.get('data_manager', {})
        self.cache: Dict[str, pd.DataFrame] = {}
        self.cache_expiry_minutes = self.config.get('cache_expiry_minutes', 60)
        
        logger.info(f"DataManager initialized with source: yfinance")

    def get_historical_data(
        self, 
        symbol: str, 
        start_date: pd.Timestamp, 
        end_date: pd.Timestamp
    ) -> pd.DataFrame:
        """
        Fetches historical OHLCV data for a given symbol and date range.
        
        Args:
            symbol (str): The ticker symbol (e.g., "AAPL").
            start_date (pd.Timestamp): The start of the date range.
            end_date (pd.Timestamp): The end of the date range.
            
        Returns:
            pd.DataFrame: A DataFrame with OHLCV data, indexed by Timestamp.
                          Returns an empty DataFrame on failure.
        """
        
        # 1. Check cache (simplified cache - does not check date range)
        # A real cache would be more granular (e.g., symbol + date_range)
        if symbol in self.cache:
            # TODO: Add expiry logic
            logger.debug(f"Returning cached data for {symbol}")
            cached_data = self.cache[symbol]
            return cached_data.loc[start_date:end_date]

        # 2. Fetch from source (yfinance)
        try:
            logger.info(f"Fetching yfinance data for {symbol} from {start_date} to {end_date}")
            ticker = yf.Ticker(symbol)
            
            # yfinance start is inclusive, end is exclusive. Add 1 day to end_date.
            data = ticker.history(
                start=start_date.strftime('%Y-%m-%d'),
                end=(end_date + timedelta(days=1)).strftime('%Y-%m-%d')
            )
            
            if data.empty:
                logger.warning(f"No data returned from yfinance for {symbol}")
                return pd.DataFrame()

            # 3. Clean and Standardize Data
            data.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }, inplace=True)
            
            # Ensure index is timezone-naive (or standardize to UTC)
            data.index = data.index.tz_localize(None)
            
            # Keep only standard columns
            standard_cols = ['open', 'high', 'low', 'close', 'volume']
            data = data[standard_cols]
            
            # 4. Store in cache
            self.cache[symbol] = data
            
            # 5. Return the requested slice
            return data.loc[start_date:end_date]

        except Exception as e:
            logger.error(f"Failed to fetch yfinance data for {symbol}: {e}", exc_info=True)
            return pd.DataFrame()

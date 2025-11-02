from typing import List, Dict, Any, Iterator, Optional
import pandas as pd
from datetime import datetime
from core.schemas.data_schema import MarketData, NewsData
from monitor.logging import get_logger

logger = get_logger(__name__)

class DataIterator:
    """
    An iterator that yields data batches for backtesting.
    It simulates a live data feed by stepping through multiple
    time-aligned data sources (e.g., market data, news).
    """

    def __init__(
        self,
        market_data_dfs: Dict[str, pd.DataFrame],
        news_data_df: Optional[pd.DataFrame] = None,
        # ... other data sources
    ):
        """
        Initializes the iterator with all data sources.
        DataFrames are expected to have a 'timestamp' column.
        
        Args:
            market_data_dfs (Dict[str, pd.DataFrame]): A dict of {symbol: DataFrame}.
            news_data_df (Optional[pd.DataFrame]): A DataFrame of news.
        """
        
        if not market_data_dfs:
            logger.error("DataIterator requires at least one market data DataFrame.")
            raise ValueError("No market data provided.")
            
        self.market_data_dfs = market_data_dfs
        self.news_data_df = news_data_df
        
        # Combine all timestamps to create a master index
        all_timestamps = set()
        for df in market_data_dfs.values():
            all_timestamps.update(df['timestamp'])
            
        if news_data_df is not None:
            all_timestamps.update(news_data_df['timestamp'])
            
        if not all_timestamps:
            logger.error("No timestamps found in any data source.")
            raise ValueError("Data sources are empty.")
            
        self.master_index = sorted(list(all_timestamps))
        self.current_step = 0
        self.total_steps = len(self.master_index)
        
        # Create iterators for each DataFrame for efficient lookup
        self.market_iterators = {
            symbol: df.set_index('timestamp').iterrows()
            for symbol, df in market_data_dfs.items()
        }
        self.news_iterator = (
            news_data_df.set_index('timestamp').iterrows()
            if news_data_df is not None else None
        )
        
        # Pre-load the first item to handle timestamp alignment
        # This is a simplified approach. A real implementation would be
        # more complex, handling `iterrows` exhaustion.
        
        # A simpler (but less memory-efficient for huge data) approach:
        # Set index on all DFs
        self.market_data_indexed = {
            symbol: df.set_index('timestamp')
            for symbol, df in market_data_dfs.items()
        }
        self.news_data_indexed = (
            news_data_df.set_index('timestamp')
            if news_data_df is not None else None
        )
        
        logger.info(f"DataIterator initialized with {self.total_steps} unique timestamps.")

    def __iter__(self) -> "DataIterator":
        self.current_step = 0
        return self

    def __next__(self) -> Dict[str, Any]:
        """
        Yields the data batch for the next timestamp in the master index.
        """
        if self.current_step >= self.total_steps:
            raise StopIteration
            
        current_timestamp = self.master_index[self.current_step]
        
        batch = {
            "timestamp": current_timestamp,
            "market_data": [],
            "news_data": []
        }
        
        # Get market data for this timestamp
        for symbol, df_indexed in self.market_data_indexed.items():
            if current_timestamp in df_indexed.index:
                row = df_indexed.loc[current_timestamp]
                # Handle multiple rows for same timestamp (unlikely in master index)
                if isinstance(row, pd.Series):
                    market_data = self._row_to_market_data(row, symbol, current_timestamp)
                    batch["market_data"].append(market_data)
                # else: Handle DataFrame rows
                    
        # Get news data for this timestamp
        if self.news_data_indexed is not None:
             if current_timestamp in self.news_data_indexed.index:
                rows = self.news_data_indexed.loc[current_timestamp]
                if isinstance(rows, pd.Series):
                    # Single news item
                    news_data = self._row_to_news_data(rows, current_timestamp)
                    batch["news_data"].append(news_data)
                elif isinstance(rows, pd.DataFrame):
                    # Multiple news items at exact same timestamp
                    for _, row in rows.iterrows():
                        news_data = self._row_to_news_data(row, current_timestamp)
                        batch["news_data"].append(news_data)

        self.current_step += 1
        return batch

    def _row_to_market_data(self, row: pd.Series, symbol: str, timestamp: datetime) -> MarketData:
        """Helper to convert a DataFrame row to a MarketData Pydantic model."""
        return MarketData(
            symbol=symbol,
            timestamp=timestamp,
            open=row.get("open", 0.0),
            high=row.get("high", 0.0),
            low=row.get("low", 0.0),
            close=row.get("close", 0.0),
            volume=row.get("volume", 0)
        )
        
    def _row_to_news_data(self, row: pd.Series, timestamp: datetime) -> NewsData:
        """Helper to convert a DataFrame row to a NewsData Pydantic model."""
        return NewsData(
            timestamp=timestamp,
            headline=row.get("headline", ""),
            source=row.get("source", "Unknown"),
            summary=row.get("summary", ""),
            url=row.get("url", ""),
            content=row.get("content", "")
        )

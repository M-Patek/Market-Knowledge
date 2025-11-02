"""
Data Iterator for Backtesting Engine.

This module provides a flexible way to iterate over historical data,
simulating the passage of time and yielding data points (e.g.,
MarketEvents, TickerData) as they would have occurred.
"""
import logging
import pandas as pd
from typing import Iterator, Union, List, Optional
from datetime import datetime
import json  # 修复：添加 json 导入

from ..core.schemas.data_schema import TickerData, MarketEvent

logger = logging.getLogger(__name__)

class DataIterator:
    """
    Iterates over one or more time-indexed data sources (e.g., CSV, Parquet)
    and yields data points in chronological order.
    """
    
    def __init__(self, file_paths: List[str], data_types: List[str], 
                 start_date: datetime, end_date: datetime,
                 timestamp_col: str = 'timestamp'):
        """
        Initializes the DataIterator.

        Args:
            file_paths: List of paths to the data files.
            data_types: List of data types corresponding to each file 
                        (e.g., 'ticker', 'market_event').
            start_date: The simulation start date.
            end_date: The simulation end date.
            timestamp_col: The name of the timestamp column in the files.
        """
        self.file_paths = file_paths
        self.data_types = data_types
        self.start_date = start_date
        self.end_date = end_date
        self.timestamp_col = timestamp_col
        self.combined_data = None
        
        if len(file_paths) != len(data_types):
            raise ValueError("file_paths and data_types must have the same length.")
            
        logger.info(f"DataIterator initialized for {len(file_paths)} sources.")

    def setup(self):
        """
        Loads all data sources, combines them, sorts by time, and filters by date.
        This is a one-time setup operation.
        """
        all_dfs = []
        
        for file_path, data_type in zip(self.file_paths, self.data_types):
            try:
                logger.info(f"Loading data from {file_path} as type '{data_type}'...")
                # Simple loader, assumes CSV for this example
                # In a real scenario, this would support parquet, feather, etc.
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                elif file_path.endswith('.jsonl'):
                    records = []
                    with open(file_path, 'r') as f:
                        for line in f:
                            # 修复：使用 json.loads
                            records.append(json.loads(line))
                    df = pd.DataFrame(records)
                else:
                    logger.warning(f"Unsupported file type: {file_path}. Skipping.")
                    continue
                    
                df[self.timestamp_col] = pd.to_datetime(df[self.timestamp_col])
                df['data_type'] = data_type
                df['source_file'] = file_path
                all_dfs.append(df)
                
            except Exception as e:
                logger.error(f"Failed to load data from {file_path}: {e}", exc_info=True)
                
        if not all_dfs:
            logger.error("No data could be loaded. Cannot proceed.")
            self.combined_data = pd.DataFrame()
            return

        # Combine, filter, and sort
        logger.info("Combining and sorting all data sources...")
        self.combined_data = pd.concat(all_dfs, ignore_index=True)
        self.combined_data.sort_values(by=self.timestamp_col, inplace=True)
        
        # Filter by date range
        self.combined_data = self.combined_data[
            (self.combined_data[self.timestamp_col] >= self.start_date) &
            (self.combined_data[self.timestamp_col] <= self.end_date)
        ]
        
        logger.info(f"Setup complete. {len(self.combined_data)} total data points to iterate.")

    def _parse_row(self, row: pd.Series) -> Optional[Union[TickerData, MarketEvent]]:
        """
        Parses a row from the combined DataFrame into a Pydantic schema.
        """
        data_type = row['data_type']
        row_dict = row.to_dict()
        
        try:
            if data_type == 'ticker':
                return TickerData(
                    symbol=row_dict.get('symbol'),
                    timestamp=row_dict.get(self.timestamp_col),
                    open=row_dict.get('open'),
                    high=row_dict.get('high'),
                    low=row_dict.get('low'),
                    close=row_dict.get('close'),
                    volume=row_dict.get('volume')
                )
            elif data_type == 'market_event':
                return MarketEvent(**row_dict)
            else:
                logger.warning(f"Unknown data_type in iterator: {data_type}")
                return None
        except Exception as e:
            logger.error(f"Failed to parse row into schema {data_type}: {e}. Row: {row_dict}")
            return None

    def __iter__(self) -> Iterator[Union[TickerData, MarketEvent]]:
        """
        Returns the iterator object.
        """
        if self.combined_data is None:
            logger.info("Running one-time setup...")
            self.setup()
            
        self.current_index = 0
        return self

    def __next__(self) -> Union[TickerData, MarketEvent]:
        """
        Yields the next data point in chronological order.
        """
        if self.combined_data is None:
            raise StopIteration("Data not loaded. Call setup() first or use in a for loop.")
            
        if self.current_index < len(self.combined_data):
            row = self.combined_data.iloc[self.current_index]
            self.current_index += 1
            
            parsed_item = self._parse_row(row)
            if parsed_item:
                return parsed_item
            else:
                # Skip bad rows and try to get the next one
                return self.__next__()
        else:
            logger.info("DataIterator reached the end.")
            raise StopIteration

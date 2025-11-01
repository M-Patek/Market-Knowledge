import pandas as pd
from typing import Dict, Any, List, AsyncGenerator, Tuple

from ..monitor.logging import get_logger
from ..core.schemas.data_schema import MarketEvent, TickerData
from ..ai.data_adapter import DataAdapter

logger = get_logger(__name__)

class DataIterator:
    """
    (Backtesting) Responsible for loading, merging, and iterating through
    historical data (both market data and news events) in chronological order.
    """

    def __init__(self, config: Dict[str, Any], data_adapter: DataAdapter):
        """
        Initializes the DataIterator.
        
        Args:
            config (Dict, Any): The main strategy configuration.
            data_adapter (DataAdapter): Used to standardize raw data from files.
        """
        self.config = config.get('data_iterator', {})
        self.data_adapter = data_adapter
        
        # Paths to data files
        self.market_data_path = self.config.get('market_data_path') # e.g., "data/market_data.csv"
        self.event_data_path = self.config.get('event_data_path')   # e.g., "data/news_events.jsonl"
        
        self.start_date: pd.Timestamp = None
        self.end_date: pd.Timestamp = None
        
        self.merged_data: pd.DataFrame = None
        logger.info("DataIterator initialized.")

    def setup(self, start_date: pd.Timestamp, end_date: pd.Timestamp):
        """
        Loads, merges, and filters all data for the backtest period.
        
        Args:
            start_date: The start of the backtest.
            end_date: The end of the backtest.
        """
        self.start_date = start_date
        self.end_date = end_date
        logger.info(f"Setting up DataIterator from {start_date} to {end_date}")

        try:
            # 1. Load Market Data (e.g., CSV)
            # This is a placeholder; a real implementation would
            # load data for all assets in the universe.
            market_df = pd.read_csv(
                self.market_data_path, 
                parse_dates=['timestamp']
            )
            market_df = market_df.set_index('timestamp')
            market_df['type'] = 'TickerData'
            
            # 2. Load Event Data (e.g., JSONL)
            event_records = []
            with open(self.event_data_path, 'r') as f:
                for line in f:
                    event_records.append(json.loads(line))
            
            event_df = pd.DataFrame(event_records)
            event_df['timestamp'] = pd.to_datetime(event_df['created_at'])
            event_df = event_df.set_index('timestamp')
            event_df['type'] = 'MarketEvent'
            
            # 3. Merge and Sort
            # We merge based on timestamp index
            # This is a simplified merge; a real one is more complex
            self.merged_data = pd.concat([market_df, event_df])
            self.merged_data = self.merged_data.sort_index()
            
            # 4. Filter by Date Range
            self.merged_data = self.merged_data.loc[self.start_date:self.end_date]
            
            if self.merged_data.empty:
                raise ValueError("No data found for the specified backtest range.")
                
            logger.info(f"DataIterator setup complete. {len(self.merged_data)} total events loaded.")

        except Exception as e:
            logger.error(f"Failed to setup DataIterator: {e}", exc_info=True)
            raise

    async def __aiter__(self) -> AsyncGenerator[Tuple[pd.Timestamp, List[Dict]], None]:
        """
        Asynchronously yields data, grouped by timestamp.
        
        Yields:
            (pd.Timestamp, List[Dict]): A tuple of (timestamp, list_of_events_at_this_time)
        """
        if self.merged_data is None:
            logger.error("DataIterator not setup. Call setup() before iterating.")
            return

        # Group all events by their timestamp (e.g., by day)
        # The groupby operation iterates in chronological order
        for timestamp, group in self.merged_data.groupby(self.merged_data.index):
            
            # Convert the DataFrame rows back to dicts
            events_batch = group.to_dict('records')
            
            # (Optional) Adapt the data using the DataAdapter
            # This standardizes the data just-in-time
            standardized_batch = []
            for raw_event in events_batch:
                if raw_event['type'] == 'MarketEvent':
                    # We pass the dict representation for adaptation
                    evt = self.data_adapter.adapt_news_event(raw_event)
                    if evt: standardized_batch.append(evt.model_dump())
                elif raw_event['type'] == 'TickerData':
                    # TODO: This logic is flawed, TickerData needs to be reconstructed
                    # from the raw CSV row, not adapted.
                    # For now, just pass the dict.
                    standardized_batch.append(raw_event)
            
            if standardized_batch:
                yield timestamp, standardized_batch
                
            # Simulate async time passage (e.g., for live-like backtesting)
            await asyncio.sleep(0) # Yield control back to the event loop

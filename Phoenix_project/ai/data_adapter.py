"""
Adapts raw data streams (e.g., from websockets, APIs) into standardized
Pydantic schemas defined in core.schemas.data_schema.

This acts as an anti-corruption layer, ensuring that the rest of the
system consumes clean, validated, and consistent data structures.
"""
import logging
from typing import Union, Optional, Dict, Any
from pydantic import ValidationError

# 修复：添加 TickerData 并使用正确的相对导入
from ..core.schemas.data_schema import MarketEvent, EconomicEvent, TickerData

logger = logging.getLogger(__name__)

class DataAdapter:
    """
    Transforms raw data payloads (dicts, JSON strings) into standardized
    Pydantic models.
    """

    def __init__(self):
        """
        Initializes the DataAdapter.
        """
        self.schema_map = {
            "market_event": MarketEvent,
            "economic_event": EconomicEvent,
            "ticker_data": TickerData,
        }
        logger.info("DataAdapter initialized.")

    def parse_data(self, raw_data: Dict[str, Any], data_type: str) -> Optional[Union[MarketEvent, EconomicEvent, TickerData]]:
        """
        Parses a raw data dictionary into a specified Pydantic schema.

        Args:
            raw_data: The raw data dictionary.
            data_type: The key for the target schema (e.g., 'market_event').

        Returns:
            A Pydantic model instance if parsing is successful, None otherwise.
        """
        schema = self.schema_map.get(data_type)
        
        if not schema:
            logger.warning(f"No schema found for data_type: {data_type}")
            return None
            
        try:
            # Attempt to parse and validate the raw data
            # This is where Pydantic works its magic
            validated_data = schema(**raw_data)
            return validated_data
            
        except ValidationError as e:
            logger.error(f"Data validation failed for data_type {data_type}. Error: {e}. Raw data: {str(raw_data)[:200]}...")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during data parsing for {data_type}. Error: {e}")
            return None

    def adapt_market_event(self, raw_event: Dict[str, Any]) -> Optional[MarketEvent]:
        """
        Helper method to specifically adapt a market event.
        
        This might include more complex logic, like field remapping,
        before validation.
        """
        # Example: Simple remapping if fields didn't match
        # if 'title' in raw_event and 'headline' not in raw_event:
        #     raw_event['headline'] = raw_event.pop('title')
            
        return self.parse_data(raw_event, "market_event")

    def adapt_ticker_data(self, raw_tick: Dict[str, Any]) -> Optional[TickerData]:
        """
        Helper method to specifically adapt a raw ticker data point.
        """
        # Business logic for adaptation can go here
        # e.g., converting units, parsing non-standard timestamps
        return self.parse_data(raw_tick, "ticker_data")

# Example Usage (if run directly for testing)
if __name__ == "__main__":
    import datetime
    
    logging.basicConfig(level=logging.INFO)
    
    adapter = DataAdapter()
    
    # 1. Test MarketEvent
    raw_news = {
        "event_id": "news_12345",
        "timestamp": datetime.datetime.now().isoformat(),
        "source": "Test Source",
        "headline": "This is a test headline",
        "content": "Full content of the test news.",
        "symbols": ["TEST", "TICKER"],
        "metadata": {"sentiment": 0.8}
    }
    
    market_event = adapter.adapt_market_event(raw_news)
    if market_event:
        logger.info(f"Successfully adapted MarketEvent: {market_event.headline}")
        logger.info(market_event.json())
        
    # 2. Test TickerData (NEW)
    raw_tick = {
        "symbol": "TEST",
        "timestamp": datetime.datetime.now().isoformat(),
        "open": 100.0,
        "high": 101.0,
        "low": 99.5,
        "close": 100.5,
        "volume": 10000
    }
    
    ticker_data = adapter.adapt_ticker_data(raw_tick)
    if ticker_data:
        logger.info(f"Successfully adapted TickerData for: {ticker_data.symbol}")
        logger.info(ticker_data.json())

    # 3. Test Failed Validation
    raw_bad_news = {
        "event_id": "news_678",
        "source": "Bad Source",
        # 'timestamp' is missing (required)
        "headline": "This will fail"
    }
    
    bad_event = adapter.adapt_market_event(raw_bad_news)
    if not bad_event:
        logger.info("Successfully caught validation error for bad market event.")

import pandas as pd
import logging
from typing import Dict, Any, Union

from ..core.schemas.data_schema import MarketEvent, TickerData

logger = logging.getLogger(__name__)

class DataAdapter:
    """
    Adapts heterogeneous data inputs (e.g., news, market data) into a standardized
    format (MarketEvent, TickerData schemas) for the AI pipeline.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.seen_event_ids = set()

    def adapt_market_data(self, raw_data: Dict[str, Any]) -> Union[TickerData, None]:
        """
        Adapts raw market data (e.g., from AlphaVantage) to the TickerData schema.
        """
        try:
            # Example adaptation logic
            # This needs to be customized based on the actual raw_data structure
            if "Time Series (Daily)" in raw_data:
                time_series = raw_data["Time Series (Daily)"]
                # Get the most recent date
                recent_date = sorted(time_series.keys(), reverse=True)[0]
                data = time_series[recent_date]
                
                ticker_data = TickerData(
                    symbol=raw_data["Meta Data"]["2. Symbol"],
                    timestamp=pd.to_datetime(recent_date),
                    open=float(data["1. open"]),
                    high=float(data["2. high"]),
                    low=float(data["3. low"]),
                    close=float(data["4. close"]),
                    volume=int(data["5. volume"]),
                    source="AlphaVantage"
                )
                return ticker_data
            
            logger.warning(f"Failed to adapt market data: Unknown format {raw_data.keys()}")
            return None

        except Exception as e:
            self._handle_error(f"Error adapting market data for {raw_data.get('Meta Data', {}).get('2. Symbol')}", e)
            return None

    def adapt_news_event(self, raw_event: Dict[str, Any]) -> Union[MarketEvent, None]:
        """
        Adapts raw news data (e.g., from Benzinga) to the MarketEvent schema.
        Includes basic deduplication.
        """
        try:
            event_id = raw_event.get('id') or raw_event.get('uuid')
            
            # Deduplication check
            if not event_id or event_id in self.seen_event_ids:
                logger.info(f"Skipping duplicate or invalid event: {event_id}")
                return None
            
            self.seen_event_ids.add(event_id)

            # Example adaptation logic for Benzinga
            if 'title' in raw_event and 'created_at' in raw_event:
                market_event = MarketEvent(
                    event_id=event_id,
                    timestamp=pd.to_datetime(raw_event['created_at']),
                    source=raw_event.get('source', 'Benzinga'),
                    headline=raw_event['title'],
                    summary=raw_event.get('body', ''),
                    symbols=self._extract_symbols(raw_event.get('stocks', [])),
                    url=raw_event.get('url'),
                    metadata={
                        'authors': [author['name'] for author in raw_event.get('authors', [])],
                        'primary_symbol': raw_event.get('primary_symbol'),
                        'tags': [tag['name'] for tag in raw_event.get('tags', [])]
                    }
                )
                return market_event

            logger.warning(f"Failed to adapt news event: Missing required fields {raw_event.keys()}")
            return None

        except Exception as e:
            self._handle_error(f"Error adapting news event {raw_event.get('id')}", e)
            return None

    def _extract_symbols(self, stocks_data: List[Dict[str, Any]]) -> List[str]:
        """Utility to extract ticker symbols from nested stock data."""
        symbols = []
        for stock in stocks_data:
            if isinstance(stock, dict) and 'symbol' in stock:
                symbols.append(stock['symbol'])
        return list(set(symbols)) # Return unique symbols

    def _handle_error(self, message: str, error: Exception):
        """Centralized error logging."""
        logger.error(f"{message}: {error}", exc_info=True)

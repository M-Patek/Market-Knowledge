import asyncio
import json
import redis.asyncio as redis
from typing import Dict, Any, List, Optional
import aiohttp

from ..monitor.logging import get_logger
from .event_distributor import EventDistributor
from .risk_filter import RiskFilter
from ..ai.data_adapter import DataAdapter
from ..core.schemas.data_schema import MarketEvent, TickerData

logger = get_logger(__name__)

class RealTimeDQM:
    """A simple real-time Data Quality Monitor."""
    def __init__(self):
        self.price_cache = {}
        self.anomaly_threshold = 0.2 # 20% price jump

    def check_price_anomaly(self, ticker_data: TickerData) -> bool:
        """Checks for anomalous price jumps."""
        symbol = ticker_data.symbol
        new_price = ticker_data.close
        
        if symbol in self.price_cache:
            last_price = self.price_cache[symbol]
            if last_price > 0 and (abs(new_price - last_price) / last_price) > self.anomaly_threshold:
                logger.warning(f"DQM ANOMALY: Price jump for {symbol} "
                               f"({last_price} -> {new_price})")
                self.price_cache[symbol] = new_price
                return True # Anomaly detected
        
        self.price_cache[symbol] = new_price
        return False # No anomaly

class StreamProcessor:
    """
    Connects to real-time data streams (e.g., Benzinga, AlphaVantage)
    and processes incoming data.
    
    It performs:
    1. Adaptation (Raw JSON -> Pydantic Schema)
    2. Deduplication (using Redis)
    3. Data Quality Monitoring (DQM)
    4. Risk Filtering (fast, synchronous check)
    5. Distribution (hands off high-value events to EventDistributor)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        event_distributor: EventDistributor,
        risk_filter: RiskFilter,
        data_adapter: DataAdapter
    ):
        self.config = config.get('stream_processor', {})
        self.event_distributor = event_distributor
        self.risk_filter = risk_filter
        self.data_adapter = data_adapter
        
        self.dqm = RealTimeDQM()
        
        # Redis client for deduplication
        self.redis_client = redis.from_url(
            self.config.get('redis_url', 'redis://localhost:6379'),
            decode_responses=True
        )
        self.dedup_key_prefix = "stream_dedup:"
        self.dedup_expiry_sec = self.config.get('dedup_expiry_sec', 3600) # 1 hour
        
        # Stream sources
        self.benzinga_ws_url = self.config.get('benzinga_ws_url')
        # ... other stream configs ...
        
        self._is_running = False

    async def run(self):
        """Starts all configured stream listeners."""
        self._is_running = True
        logger.info("Starting StreamProcessor...")
        
        tasks = []
        if self.benzinga_ws_url:
            tasks.append(asyncio.create_task(
                self._listen_to_benzinga(self.benzinga_ws_url)
            ))
            
        # TODO: Add tasks for other streams (AlphaVantage, etc.)
        
        if not tasks:
            logger.warning("No data streams configured. StreamProcessor will be idle.")
            return

        await asyncio.gather(*tasks)

    def stop(self):
        """Signals the processor to stop."""
        logger.info("StreamProcessor received stop signal.")
        self._is_running = False
        # (The websocket tasks must handle this signal)

    async def _listen_to_benzinga(self, ws_url: str):
        """Main connection loop for a Benzinga websocket."""
        logger.info(f"Connecting to Benzinga stream: {ws_url}")
        
        while self._is_running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(ws_url) as ws:
                        
                        # TODO: Add authentication logic (e.g., sending auth message)
                        
                        async for msg in ws:
                            if not self._is_running:
                                break
                                
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                await self._process_raw_message(msg.data, "Benzinga")
                            elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                                logger.warning(f"Benzinga websocket closed/errored: {msg.data}")
                                break
            
            except Exception as e:
                logger.error(f"Benzinga stream connection failed: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5) # Reconnect delay

    async def _process_raw_message(self, raw_data: str, source: str):
        """Handles a single raw message from any stream."""
        try:
            raw_event = json.loads(raw_data)
            
            # --- 1. Adaptation ---
            # Standardize the data (e.g., Benzinga -> MarketEvent)
            # We assume a 'type' field exists in the raw data
            event_type = raw_event.get('type', 'unknown')
            
            event_object: Optional[Union[MarketEvent, TickerData]] = None
            
            if event_type == 'news':
                event_object = self.data_adapter.adapt_news_event(raw_event)
            elif event_type == 'price':
                event_object = self.data_adapter.adapt_market_data(raw_event)
            else:
                logger.debug(f"Unknown event type received from {source}: {event_type}")
                return

            if not event_object:
                # Adaptation failed or was a duplicate
                return

            # --- 2. Deduplication (using adapted event_id) ---
            if isinstance(event_object, MarketEvent):
                dedup_key = f"{self.dedup_key_prefix}{event_object.event_id}"
                if await self.redis_client.set(dedup_key, "1", ex=self.dedup_expiry_sec, nx=True) is None:
                    logger.debug(f"Discarding duplicate event: {event_object.event_id}")
                    return
            
            # --- 3. Data Quality Monitoring (DQM) ---
            if isinstance(event_object, TickerData):
                if self.dqm.check_price_anomaly(event_object):
                    # TODO: Log anomaly, but decide whether to discard
                    pass
                # Price data is usually not "published" to the orchestrator,
                # but rather just updates the ContextBus state.
                # (This logic is simplified)
                # await self.context_bus.update_market_data(event_object)
                return

            # --- 4. High-Speed Risk Filtering ---
            if isinstance(event_object, MarketEvent):
                risk_match = self.risk_filter.check_event(event_object)
                if risk_match:
                    event_object.metadata['risk_filter_match'] = risk_match
                    # (All risk events are high-value by default)
                
                # TODO: Add 'value' filter (e.g., based on symbols, tags)
                is_high_value = True # For now, assume all news is high-value
                
                # --- 5. Distribution ---
                if is_high_value:
                    # Send the standardized, high-value event to the
                    # processing queue.
                    await self.event_distributor.publish(event_object)

        except json.JSONDecodeError:
            logger.warning(f"Received invalid JSON from {source}: {raw_data[:100]}...")
        except Exception as e:
            logger.error(f"Error processing raw message: {e}", exc_info=True)

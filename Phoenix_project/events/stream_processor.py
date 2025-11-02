"""
Real-time Event Stream Processor.

Connects to live data streams (e.g., WebSocket), validates data,
and pushes standardized events to the EventDistributor (Redis queue).
"""
import asyncio
import websockets
import json
import logging
import os
import pandas as pd
from typing import Optional

from .event_distributor import EventDistributor
# 修复：添加 TickerData 并使用正确的相对导入
from ..core.schemas.data_schema import MarketEvent, TickerData
from ..ai.data_adapter import DataAdapter
from .risk_filter import RiskFilter

logger = logging.getLogger(__name__)

class StreamProcessor:
    """
    Connects to a WebSocket stream, processes messages, and distributes them.
    """
    
    def __init__(self, 
                 stream_uri: str, 
                 distributor: EventDistributor, 
                 adapter: DataAdapter,
                 risk_filter: RiskFilter):
        """
        Initializes the StreamProcessor.

        Args:
            stream_uri: The WebSocket URI to connect to.
            distributor: The EventDistributor to push events to.
            adapter: The DataAdapter to standardize data.
            risk_filter: The RiskFilter to check events.
        """
        self.stream_uri = stream_uri
        self.distributor = distributor
        self.adapter = adapter
        self.risk_filter = risk_filter
        self.running = False
        logger.info(f"StreamProcessor initialized for URI: {stream_uri}")

    async def _process_message(self, raw_message: str):
        """
        Processes a single raw message from the WebSocket.
        """
        try:
            msg = json.loads(raw_message)
            
            # --- Example: Differentiating between stream types ---
            
            # 1. Market Event (News) Stream
            # (Assuming a hypothetical news stream format)
            if msg.get('stream_type') == 'news':
                # Use adapter to standardize
                event: Optional[MarketEvent] = self.adapter.adapt_market_event(msg.get('data', {}))
                
                if not event:
                    logger.warning(f"Failed to adapt news message: {str(raw_message)[:100]}...")
                    return

                # Check against the Risk Filter
                if self.risk_filter.is_high_risk(event.headline) or self.risk_filter.is_high_risk(event.content):
                    event.metadata['risk_filter_triggered'] = True
                    logger.warning(f"High-risk event detected: {event.headline}")
                    # Optionally, route high-risk events to a different queue
                    # await self.distributor.push_event(event, queue_name="high_risk_queue")
                
                # Push to the standard event distributor
                await self.distributor.push_event(event)
                logger.debug(f"Distributed MarketEvent: {event.event_id}")

            # 2. Ticker Data Stream
            # (Assuming a format like Polygon.io or Binance)
            elif msg.get('e') == 'k': # 'e' for event type, 'k' for kline/candlestick
                kline = msg.get('k', {})
                # Adapt raw kline data to our standardized TickerData schema
                # 修复：使用 TickerData schema
                data = TickerData(
                    symbol=kline.get('s'),
                    timestamp=pd.to_datetime(kline.get('T'), unit='ms'), # Kline start time
                    open=float(kline.get('o')),
                    high=float(kline.get('h')),
                    low=float(kline.get('l')),
                    close=float(kline.get('c')),
                    volume=float(kline.get('v'))
                )
                
                # Note: TickerData might not go to the main event queue.
                # It might go to a different queue or be handled directly
                # for real-time feature generation.
                # For this example, we'll assume it's pushed for logging/monitoring.
                await self.distributor.push_event(data, queue_name="ticker_queue")
                logger.debug(f"Distributed TickerData: {data.symbol}")

            else:
                logger.debug(f"Received unhandled message type: {str(raw_message)[:100]}...")
                
        except json.JSONDecodeError:
            logger.warning(f"Received non-JSON message: {raw_message}")
        except Exception as e:
            logger.error(f"Error processing message: {e}. Message: {raw_message}", exc_info=True)

    async def start(self):
        """
        Starts the WebSocket client and message processing loop.
        """
        self.running = True
        logger.info(f"Attempting to connect to WebSocket: {self.stream_uri}")
        
        while self.running:
            try:
                async with websockets.connect(self.stream_uri) as websocket:
                    logger.info(f"Successfully connected to {self.stream_uri}")
                    async for message in websocket:
                        if not self.running:
                            break
                        await self._process_message(message)
                        
            except websockets.exceptions.ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed (code={e.code}, reason={e.reason}). Retrying in 5s...")
            except Exception as e:
                logger.error(f"WebSocket error: {e}. Retrying in 5s...")
                
            if self.running:
                await asyncio.sleep(5) # Wait before retrying

    def stop(self):
        """
        Stops the WebSocket client.
        """
        self.running = False
        logger.info("Stopping StreamProcessor...")

# Example usage (requires a running Redis)
if __name__ == "__main__":
    
    # This example won't connect to a real stream unless one is provided,
    # but it demonstrates the setup.
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Load a mock stream URI
    STREAM_URI = os.environ.get("MOCK_STREAM_URI", "ws://echo.websocket.org")
    REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost")

    async def main():
        # Setup dependencies
        try:
            distributor = EventDistributor(redis_url=REDIS_URL)
            await distributor.connect()
        except Exception as e:
            logger.error(f"Could not connect to Redis at {REDIS_URL}. Aborting. Error: {e}")
            return
            
        adapter = DataAdapter()
        risk_filter = RiskFilter(config_path="config/event_filter_config.yaml")
        
        processor = StreamProcessor(
            stream_uri=STREAM_URI,
            distributor=distributor,
            adapter=adapter,
            risk_filter=risk_filter
        )
        
        try:
            logger.info("Starting StreamProcessor...")
            # For testing, just run for 10 seconds
            # In production, this would be `await processor.start()`
            await asyncio.wait_for(processor.start(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.info("Test run finished.")
        except KeyboardInterrupt:
            logger.info("Manual interruption.")
        finally:
            processor.stop()
            await distributor.disconnect()
            logger.info("StreamProcessor shut down.")

    # To run this example:
    # 1. Make sure you have a Redis server running on localhost.
    # 2. You can use 'ws://echo.websocket.org' to test connectivity.
    #    It will just echo back whatever you send (which won't parse, but tests the loop).
    #
    # asyncio.run(main())
    logger.info("StreamProcessor example complete. Run `asyncio.run(main())` to test.")

"""
Data Producer for Phoenix.

OPTIMIZED: This module is no longer a simulation.
It connects to a real-time WebSocket feed (e.g., Polygon.io)
and produces live market data into the Redis stream (or Kafka).
This replaces the previous 'data_producer.py' simulation.
"""

import json
import logging
import os
import threading
import time
from typing import Any, Dict

import redis  # type: ignore
import websocket  # type: ignore  (需要 'pip install websocket-client')

from .core.schemas.data_schema import MarketData
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class RealTimeDataProducer:
    """
    Connects to a real-time data provider (WebSocket) and streams
    data into the system's message bus (Redis Streams).
    """

    def __init__(self, config: Dict[str, Any], redis_client: redis.Redis):
        """
        Initializes the RealTimeDataProducer.

        Args:
            config: Configuration dictionary. Must contain 'data_producer'
                    section with 'websocket_url' and 'api_key'.
            redis_client: An initialized Redis client.
        """
        self.config = config.get("data_producer", {})
        self.redis_client = redis_client

        # 从配置或环境变量中获取 WebSocket URL 和 API Key
        self.ws_url = self.config.get("websocket_url")
        self.api_key = self.config.get("api_key") or os.getenv("DATA_API_KEY")
        self.symbols = self.config.get("symbols", ["T:SPY", "T:QQQ"]) # 示例: Polygon.io Tickers
        
        # Redis Stream 名称
        self.stream_name = self.config.get("stream_name", "phoenix:stream:market_data")
        # 用于存储最新价格的 Redis Key 前缀
        self.live_key_prefix = "phoenix:market_data:live:"

        if not self.ws_url or not self.api_key:
            logger.error("WebSocket URL or API Key not configured. Producer cannot start.")
            raise ValueError("Missing websocket_url or api_key in config")

        self.ws: Optional[websocket.WebSocketApp] = None
        self._stop_event = threading.Event()

    def _on_message(self, ws, message):
        """Callback for handling incoming WebSocket messages."""
        try:
            data_list = json.loads(message)
            
            # 消息通常是列表
            for data in data_list:
                # --- 解析逻辑 (特定于 Polygon.io Tickers) ---
                if data.get("ev") == "T": # 这是一个 Ticker
                    symbol = data.get("sym")
                    price = data.get("p")
                    volume = data.get("s")
                    # 纳秒时间戳 -> datetime
                    timestamp = datetime.fromtimestamp(data.get("t") / 1000.0)

                    if not all([symbol, price, volume, timestamp]):
                        continue

                    # 组装 MarketData 对象
                    market_data = MarketData(
                        symbol=symbol,
                        price=float(price),
                        volume=float(volume),
                        timestamp=timestamp
                    )
                    
                    self._produce_to_redis(market_data)
                
                elif data.get("ev") == "status":
                    logger.info(f"WebSocket Status Update: {data.get('message')}")

        except json.JSONDecodeError:
            logger.warning(f"Received non-JSON message: {message}")
        except Exception as e:
            logger.error(f"Error processing message: {e}\nMessage: {message}")

    def _produce_to_redis(self, data: MarketData):
        """Pushes data into Redis Stream and updates the 'live' key."""
        try:
            data_dict = data.model_dump(mode="json")
            # model_dump(mode='json') 会将 datetime 转换为 isoformat 字符串
            
            # 1. 发布到 Stream
            self.redis_client.xadd(self.stream_name, data_dict)
            
            # 2. 更新最新价格 (live key)
            live_key = f"{self.live_key_prefix}{data.symbol}"
            self.redis_client.set(live_key, json.dumps(data_dict))
            
            logger.debug(f"Produced to {self.stream_name}: {data.symbol} @ {data.price}")

        except redis.RedisError as e:
            logger.error(f"Redis error producing data: {e}")

    def _on_error(self, ws, error):
        """Callback for WebSocket errors."""
        logger.error(f"WebSocket Error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        """Callback for WebSocket close."""
        logger.info(f"WebSocket closed. Code: {close_status_code}, Msg: {close_msg}")
        if not self._stop_event.is_set():
            logger.info("Attempting to reconnect in 5 seconds...")
            time.sleep(5)
            self.run() # 尝试重连

    def _on_open(self, ws):
        """Callback for WebSocket open. Handles authentication and subscription."""
        logger.info("WebSocket connection opened.")
        try:
            # 1. 认证
            auth_data = {
                "action": "auth",
                "params": self.api_key
            }
            ws.send(json.dumps(auth_data))
            
            # 2. 订阅
            subscribe_data = {
                "action": "subscribe",
                "params": ",".join(self.symbols) # 例如 "T:SPY,T:QQQ"
            }
            ws.send(json.dumps(subscribe_data))
            logger.info(f"Subscribed to symbols: {self.symbols}")
            
        except Exception as e:
            logger.error(f"Error during WebSocket auth/subscribe: {e}")

    def run(self):
        """
        Starts the WebSocket client in a blocking loop.
        """
        if self._stop_event.is_set():
            logger.info("Producer is marked as stopped. Will not start.")
            return

        logger.info(f"Starting real-time data producer. Connecting to {self.ws_url}...")
        
        # websocket.enableTrace(True) # 取消注释以进行详细调试
        
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
        
        # run_forever() 是一个阻塞调用
        self.ws.run_forever()

    def stop(self):
        """Stops the WebSocket client."""
        logger.info("Stopping data producer...")
        self._stop_event.set()
        if self.ws:
            self.ws.close()
        logger.info("Data producer stopped.")


# --- 用于独立运行的示例 ---
def main():
    """
    Example of how to run the producer independently.
    """
    logger.info("Starting Data Producer (Standalone Mode)...")
    
    # 从环境变量加载配置
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    WEBSOCKET_URL = os.getenv("POLYGON_WS_URL", "wss://socket.polygon.io/stocks")
    API_KEY = os.getenv("POLYGON_API_KEY")

    if not API_KEY:
        logger.critical("POLYGON_API_KEY environment variable not set. Exiting.")
        return

    try:
        redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=False)
        redis_client.ping()
        logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
    except redis.RedisError as e:
        logger.critical(f"Failed to connect to Redis: {e}. Exiting.")
        return

    config = {
        "data_producer": {
            "websocket_url": WEBSOCKET_URL,
            "api_key": API_KEY,
            "symbols": ["T:SPY", "T:QQQ", "T:AAPL", "T:MSFT"],
            "stream_name": "phoenix:stream:market_data"
        }
    }

    producer = RealTimeDataProducer(config, redis_client)
    
    try:
        producer.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        producer.stop()
        logger.info("Data Producer shut down.")

if __name__ == "__main__":
    main()

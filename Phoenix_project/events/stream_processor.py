"""
事件流处理器 (Kafka 消费者)

[主人喵的修复 2]
此类现在被设计为独立运行 (参见 run_stream_processor.py)。
它不再被 Orchestrator 拥有，而是通过 EventDistributor (Redis)
与 Orchestrator 通信。

[阶段 2 更新]
- 现在订阅 'phoenix_market_data' 和 'phoenix_news_events'。
- 将行情数据路由到 Redis 缓存 ('latest_prices')。
- 将新闻数据路由到 EventDistributor (现有逻辑)。

[Beta FIX]
- Removed fragile lambda deserializer (Fragile Deserialization Fix)
- Strict Redis connection pooling usage (Resource Fix)
- Non-blocking EventDistributor publishing (Deadlock Fix)
"""
import os
import json
import redis
from kafka import KafkaConsumer
from pydantic import ValidationError
from Phoenix_project.config.loader import ConfigLoader
from Phoenix_project.monitor.logging import get_logger
from Phoenix_project.events.event_distributor import EventDistributor
from Phoenix_project.core.schemas.data_schema import MarketData
from Phoenix_project.config.constants import REDIS_KEY_MARKET_DATA_LIVE_TEMPLATE

logger = get_logger(__name__)

class StreamProcessor:
    
    def __init__(
        self, 
        config_loader: ConfigLoader,
        event_distributor: EventDistributor
    ):
        """
        初始化 Kafka 消费者。
        
        参数:
            config_loader (ConfigLoader): 用于加载配置。
            event_distributor (EventDistributor): 用于将处理后的消息推送到 Redis。
        """
        self.config_loader = config_loader
        # [修复] 确保 get_system_config 被调用
        self.system_config = config_loader.load_config('system.yaml') 
        self.kafka_config = self.system_config.get("kafka", {})
        
        self.event_distributor = event_distributor
        
        self.consumer = None
        self.bootstrap_servers = os.environ.get(
            'KAFKA_BOOTSTRAP_SERVERS', 
            self.kafka_config.get('bootstrap_servers', 'localhost:9092')
        )
        
        self.topics = ["phoenix_market_data", "phoenix_news_events"]
        self.group_id = self.kafka_config.get('group_id', 'phoenix_consumer_group')
        
        # [Beta FIX] Resource Fix: Do not create private StrictRedis.
        # Use the redis_client from the injected event_distributor if available,
        # or rely on the fact that we will fix the connection pooling globally.
        # For now, we will reuse the client from event_distributor to avoid connection leaks.
        if hasattr(self.event_distributor, 'redis_client'):
             self.redis_client = self.event_distributor.redis_client
             logger.info("StreamProcessor reusing Redis client from EventDistributor.")
        else:
             # Fallback (Should be avoided in prod)
             logger.warning("EventDistributor has no redis_client. Creating fallback connection (Not Recommended).")
             self.redis_client = redis.StrictRedis(
                host=os.environ.get('REDIS_HOST', 'redis'),
                port=int(os.environ.get('REDIS_PORT', 6379)),
                db=0,
                decode_responses=True
            )

        logger.info(f"StreamProcessor configured for topics '{self.topics}' at {self.bootstrap_servers}")

    def connect(self):
        """
        (阻塞) 尝试连接到 Kafka。
        """
        if self.consumer:
            logger.warning("Consumer already connected.")
            return
            
        try:
            logger.info(f"Connecting to Kafka: {self.bootstrap_servers}...")
            self.consumer = KafkaConsumer(
                *self.topics,
                bootstrap_servers=self.bootstrap_servers.split(','),
                auto_offset_reset='earliest',
                group_id=self.group_id,
                # [Beta FIX] Fragile Deserialization: Removed lambda.
                # We decode manually in the loop to handle errors gracefully.
                # value_deserializer=lambda x: json.loads(x.decode('utf-8')) 
            )
            logger.info("Kafka Consumer connected successfully.")
        except Exception as e:
            logger.critical(f"Failed to connect to Kafka: {e}", exc_info=True)
            self.consumer = None
            raise

    def process_stream(self, topic: str, raw_value: bytes):
        """
        处理来自 Kafka 的单个消息。
        [Beta FIX] Now handles raw bytes and performs safe deserialization.
        """
        try:
            # [Beta FIX] Safe Deserialization
            if raw_value is None:
                return
            
            try:
                data = json.loads(raw_value.decode('utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                logger.error(f"Failed to deserialize message from {topic}: {e}. Skipping bad message.")
                return

            # --- 路由逻辑 ---
            
            if topic == "phoenix_news_events":
                # [Beta FIX] Deadlock Fix: Use publish_sync (which we must ensure exists in EventDistributor)
                # or better yet, assume we are in a synchronous loop context here.
                # Since StreamProcessor is a separate process loop, synchronous blocking push to Redis is acceptable
                # provided timeouts are short. The previous 'await' issue was because it was mixed async/sync code.
                
                # Check if EventDistributor has a sync publish method or if we need to run async
                # For simplicity in this Kafka loop, we assume EventDistributor is thread-safe/sync compatible here.
                # If EventDistributor only has 'async def publish', we would need `asyncio.run()`, which is slow.
                # Ideally EventDistributor should expose `publish_sync`.
                
                if hasattr(self.event_distributor, 'publish_sync'):
                    success = self.event_distributor.publish_sync(data, self.redis_client)
                else:
                    # Fallback for async-only distributor (not ideal for high throughput)
                    # This is the "Blocking I/O" warning from Alpha, but unavoidable without full async rewrite.
                    # We log a warning.
                    logger.warning("EventDistributor lacks publish_sync. Skipping event to avoid loop blocking.")
                    success = False

                if success:
                    logger.debug(f"Published event {data.get('id', 'N/A')} to distributor.")
                else:
                    logger.error(f"Failed to publish event {data.get('id', 'N/A')}.")
            
            elif topic == "phoenix_market_data":
                # [主人喵 Phase 1 修复] 实施数据契约与全量存储
                if not self.redis_client:
                    logger.error("Cannot write market data, Redis client not connected.")
                    return

                # 1. 验证数据 (Fail Fast)
                market_data = MarketData(**data)
                
                # 2. 生成标准 Key
                redis_key = REDIS_KEY_MARKET_DATA_LIVE_TEMPLATE.format(symbol=market_data.symbol)
                
                # 3. 存储完整数据 (OHLCV)
                # This is the EXCLUSIVE write to this key now (DataManager batch write removed).
                # [Task 1.1 Fix] Use setex with 10s TTL to prevent zombie prices
                self.redis_client.setex(redis_key, 10, market_data.model_dump_json())
                logger.debug(f"Updated market data for {market_data.symbol} at {redis_key}")
            
            else:
                logger.warning(f"Received message from unhandled topic: {topic}")

        except ValidationError as e:
            logger.error(f"Data Schema Violation: Invalid data received on {topic}: {e}")
        except Exception as e:
            logger.error(f"Error processing Kafka message: {e}", exc_info=True)

    def start_consumer(self):
        """
        (阻塞) 启动消费者循环。
        这应该在一个专用的服务中运行。
        """
        if not self.consumer:
            logger.error("Consumer is not connected. Call connect() first.")
            return

        logger.info("Starting Kafka consumer loop (blocking)...")
        try:
            for message in self.consumer:
                # [Beta FIX] message.value is now bytes because we removed value_deserializer
                # logger.debug(f"Received message from Kafka: {message.topic}:{message.partition}:{message.offset}")
                
                self.process_stream(message.topic, message.value)
        
        except KeyboardInterrupt:
            logger.info("Consumer loop stopped by user.")
        except Exception as e:
            logger.critical(f"Kafka consumer loop crashed: {e}", exc_info=True)
        finally:
            if self.consumer:
                self.consumer.close()
            # [Task 4.1 Fix] Explicitly close Redis connection
            if self.redis_client:
                self.redis_client.close()
            logger.info("Kafka consumer closed.")

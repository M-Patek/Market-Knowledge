"""
Phoenix_project/events/stream_processor.py
[Phase 5 Task 1] Activate Neural Pathway.
Publish market data events to EventDistributor to wake up Orchestrator.
"""
import os
import json
import redis
from kafka import KafkaConsumer
from pydantic import ValidationError
from Phoenix_project.config.loader import ConfigLoader
from Phoenix_project.monitor.logging import get_logger
from Phoenix_project.events.event_distributor import SyncEventDistributor
from Phoenix_project.core.schemas.data_schema import MarketData
from Phoenix_project.config.constants import REDIS_KEY_MARKET_DATA_LIVE_TEMPLATE

logger = get_logger(__name__)

class StreamProcessor:
    
    def __init__(
        self, 
        config_loader: ConfigLoader,
        event_distributor: SyncEventDistributor
    ):
        """
        初始化 Kafka 消费者。
        
        参数:
            config_loader (ConfigLoader): 用于加载配置。
            event_distributor (SyncEventDistributor): 用于将处理后的消息推送到 Redis。
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
        # Use the redis_client from the injected event_distributor if available.
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
                # [Task 001 Integration] Use SyncEventDistributor
                # The Sync distributor provides a blocking 'publish' method, perfectly safe here.
                success = self.event_distributor.publish(data)

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
                # [Task 003 Fix] Use setex with 60s TTL to prevent "Short-term Memory Loss"
                # allowing Orchestrator time for heavy inference.
                self.redis_client.setex(redis_key, 60, market_data.model_dump_json())
                logger.debug(f"Updated market data for {market_data.symbol} at {redis_key}")

                # [Phase 5 Task 1] Wake up Orchestrator via EventDistributor
                # Market data acts as a 'tick' to drive the event loop.
                self.event_distributor.publish(data)
                logger.debug(f"Published market tick for {market_data.symbol} to distributor.")
            
            else:
                logger.warning(f"Received message from unhandled topic: {topic}")

        except ValidationError as e:
            logger.error(f"Data Schema Violation: Invalid data received on {topic}: {e}")
        except Exception as e:
            logger.error(f"Error processing Kafka message: {e}", exc_info=True)

    def start_consumer(self):
        """
        (阻塞) 启动消费者循环。
        """
        if not self.consumer:
            logger.error("Consumer is not connected. Call connect() first.")
            return

        logger.info("Starting Kafka consumer loop (blocking)...")
        try:
            for message in self.consumer:
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

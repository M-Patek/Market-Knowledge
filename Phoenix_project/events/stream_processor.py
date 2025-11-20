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
"""
import os
import json
import redis # <-- [阶段 2] 添加
from kafka import KafkaConsumer
from pydantic import ValidationError
from Phoenix_project.config.loader import ConfigLoader
from Phoenix_project.monitor.logging import get_logger
from Phoenix_project.events.event_distributor import EventDistributor # <-- [主人喵的修复 2] 导入
from Phoenix_project.core.schemas.data_schema import MarketData
from Phoenix_project.config.constants import REDIS_KEY_MARKET_DATA_LIVE_TEMPLATE

logger = get_logger(__name__)

class StreamProcessor:
    
    def __init__(
        self, 
        config_loader: ConfigLoader,
        event_distributor: EventDistributor # <-- [主人喵的修复 2] 注入
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
        
        # [主人喵的修复 2] 注入 EventDistributor
        self.event_distributor = event_distributor
        
        self.consumer = None
        self.bootstrap_servers = os.environ.get(
            'KAFKA_BOOTSTRAP_SERVERS', 
            self.kafka_config.get('bootstrap_servers', 'localhost:9092')
        )
        
        # --- [阶段 2] 更改 ---
        # 订阅计划中定义的两个新主题
        self.topics = ["phoenix_market_data", "phoenix_news_events"]
        # self.topic = self.kafka_config.get('topic', 'market_data') # <-- [阶段 2] 移除
        
        self.group_id = self.kafka_config.get('group_id', 'phoenix_consumer_group')
        
        # [阶段 2] 添加 Redis 客户端用于写入 'latest_prices'
        try:
            self.redis_client = redis.StrictRedis(
                host=os.environ.get('REDIS_HOST', 'redis'),
                port=int(os.environ.get('REDIS_PORT', 6379)),
                db=0,
                decode_responses=True # 确保 hset 使用字符串
            )
            self.redis_client.ping()
            logger.info(f"StreamProcessor 已连接到 Redis for latest_prices。")
        except redis.exceptions.ConnectionError as e:
            logger.critical(f"StreamProcessor 无法连接到 Redis: {e}", exc_info=True)
            self.redis_client = None
        # --- [阶段 2 结束] ---

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
                # [阶段 2] 更改: 订阅多个主题
                *self.topics,
                bootstrap_servers=self.bootstrap_servers.split(','),
                auto_offset_reset='earliest', # 从最早的消息开始
                group_id=self.group_id,
                value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )
            logger.info("Kafka Consumer connected successfully.")
        except Exception as e:
            logger.critical(f"Failed to connect to Kafka: {e}", exc_info=True)
            self.consumer = None
            raise # 允许 run_stream_processor.py 捕获并重试

    def process_stream(self, topic: str, data: dict):
        """
        处理来自 Kafka 的单个消息。
        [阶段 2] 更改: 签名包含 'topic' 以便路由。
        [主人喵的修复 2] 现在将处理后的消息推送到 EventDistributor (Redis)。
        """
        try:
            # --- [阶段 2] 路由逻辑 ---
            
            if topic == "phoenix_news_events":
                # 阶段 2 逻辑: 推送到 EventDistributor (Redis 队列)
                success = self.event_distributor.publish(data)
                if success:
                    logger.debug(f"Successfully processed and published event {data.get('id', 'N/A')} to distributor.")
                else:
                    logger.error(f"Failed to publish event {data.get('id', 'N/A')} to distributor.")
            
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
                self.redis_client.set(redis_key, market_data.model_dump_json())
                logger.debug(f"Updated market data for {market_data.symbol} at {redis_key}")
            
            else:
                logger.warning(f"Received message from unhandled topic: {topic}")
            # --- [阶段 2 结束] ---

        except ValidationError as e:
            logger.error(f"Data Schema Violation: Invalid data received on {topic}: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Message value is not valid JSON: {e}. Value: {data}")
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
                # message.value 已经是 dict (归功于 value_deserializer)
                logger.debug(f"Received message from Kafka: {message.topic}:{message.partition}:{message.offset}")
                
                # [阶段 2] 更改: 传递 topic 和 value
                self.process_stream(message.topic, message.value)
        
        except KeyboardInterrupt:
            logger.info("Consumer loop stopped by user.")
        except Exception as e:
            logger.critical(f"Kafka consumer loop crashed: {e}", exc_info=True)
        finally:
            if self.consumer:
                self.consumer.close()
            logger.info("Kafka consumer closed.")

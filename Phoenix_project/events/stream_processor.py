"""
事件流处理器 (Kafka 消费者)

[主人喵的修复 2]
此类现在被设计为独立运行 (参见 run_stream_processor.py)。
它不再被 Orchestrator 拥有，而是通过 EventDistributor (Redis)
与 Orchestrator 通信。
"""
import os
import json
from kafka import KafkaConsumer
from Phoenix_project.config.loader import ConfigLoader
from Phoenix_project.monitor.logging import get_logger
from Phoenix_project.events.event_distributor import EventDistributor # <-- [主人喵的修复 2] 导入

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
        self.system_config = config_loader.get_system_config()
        self.kafka_config = self.system_config.get("kafka", {})
        
        # [主人喵的修复 2] 注入 EventDistributor
        self.event_distributor = event_distributor
        
        self.consumer = None
        self.bootstrap_servers = os.environ.get(
            'KAFKA_BOOTSTRAP_SERVERS', 
            self.kafka_config.get('bootstrap_servers', 'localhost:9092')
        )
        self.topic = self.kafka_config.get('topic', 'market_data')
        self.group_id = self.kafka_config.get('group_id', 'phoenix_consumer_group')
        
        logger.info(f"StreamProcessor configured for topic '{self.topic}' at {self.bootstrap_servers}")

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
                self.topic,
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

    def process_stream(self, raw_message: dict):
        """
        处理来自 Kafka 的单个消息。
        [主人喵的修复 2] 现在将处理后的消息推送到 EventDistributor (Redis)。
        """
        try:
            # (假设 raw_message 已经是反序列化后的 dict)
            
            # 1. (TODO) 在此处添加任何需要的验证或转换
            data = raw_message
            if 'id' not in data:
                logger.warning(f"Message missing 'id', discarding: {data}")
                return

            # 2. [主人喵的修复 2] 将事件发布到 Redis 队列
            success = self.event_distributor.publish(data)
            
            if success:
                logger.debug(f"Successfully processed and published event {data['id']} to distributor.")
            else:
                logger.error(f"Failed to publish event {data['id']} to distributor.")

        except json.JSONDecodeError as e:
            logger.error(f"Message value is not valid JSON: {e}. Value: {raw_message}")
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
                self.process_stream(message.value)
        
        except KeyboardInterrupt:
            logger.info("Consumer loop stopped by user.")
        except Exception as e:
            logger.critical(f"Kafka consumer loop crashed: {e}", exc_info=True)
        finally:
            if self.consumer:
                self.consumer.close()
            logger.info("Kafka consumer closed.")

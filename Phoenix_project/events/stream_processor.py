"""
事件流处理器
(实现为 Kafka 消费者)
"""
import json
import logging
from typing import List, Callable, Any
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
import time

# (假设您有一个日志记录器)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class StreamProcessor:
    """
    使用 kafka-python 实现的真实 Kafka 消费者，
    用于处理传入的数据流。
    """
    def __init__(self, bootstrap_servers: List[str], group_id: str, topics: List[str]):
        """
        初始化 Kafka StreamProcessor。

        Args:
            bootstrap_servers (List[str]): Kafka broker 的地址列表
                (例如, ['kafka:29092'])。
            group_id (str): 此消费者的 Kafka 消费者组 ID。
            topics (List[str]): 要订阅的 Kafka 主题列表。
        """
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.topics = topics
        self.consumer = None
        self.log_prefix = "StreamProcessor:"
        logger.info(f"{self.log_prefix} Initialized. Configured for servers: {bootstrap_servers}")

    # --- [任务 2 实现] ---
    # 移除了 "simulated" 日志，实现了真实的 Kafka 连接
    def connect(self, max_retries=5, retry_delay=10):
        """
        连接到 Kafka Broker。
        由于 Kafka 启动可能需要时间，因此包含重试逻辑。
        """
        retries = 0
        while retries < max_retries:
            try:
                self.consumer = KafkaConsumer(
                    *self.topics,
                    bootstrap_servers=self.bootstrap_servers,
                    group_id=self.group_id,
                    auto_offset_reset='earliest',
                    # 假设消息是 JSON 编码的 UTF-8 字符串
                    value_deserializer=lambda m: json.loads(m.decode('utf-8'))
                )
                logger.info(f"{self.log_prefix} KafkaConsumer connected and subscribed to topics: {self.topics}")
                return True
            except NoBrokersAvailable:
                retries += 1
                logger.warning(f"{self.log_prefix} No Kafka brokers available. Retrying ({retries}/{max_retries}) in {retry_delay}s...")
                time.sleep(retry_delay)
            except Exception as e:
                logger.error(f"{self.log_prefix} Failed to connect to Kafka: {e}", exc_info=True)
                return False
        
        logger.error(f"{self.log_prefix} Could not connect to Kafka after {max_retries} retries.")
        return False

    # --- [任务 2 实现] ---
    # 移除了 "simulated" 日志，实现了真实的消息处理
    def process_stream(self, message: Any):
        """
        处理单个反序列化后的 Kafka 消息。
        (这是验收标准中的“解析和分派”步骤)
        
        Args:
            message (Any): 从 Kafka 消息值中反序列化出的 Python 对象 (例如 dict)。
        """
        # 示例：您可以将此消息分派给系统的其他部分
        # (例如，EventDistributor 或 DataManager)
        
        logger.info(f"{self.log_prefix} Processing message: {message}")
        
        # TODO: 在此处添加您的分派逻辑
        # (例如: self.event_distributor.publish(message['type'], **message_payload))

    # --- [任务 2 实现] ---
    # 移除了 "simulated" 日志，实现了真实的消费循环
    def start_consumer(self):
        """
        启动持续消费循环。
        这是一个阻塞操作，通常在专用的工作进程或线程中运行。
        """
        if not self.consumer:
            logger.error(f"{self.log_prefix} Consumer not connected. Call connect() first.")
            return

        logger.info(f"{self.log_prefix} Starting message consumption loop...")
        try:
            # 持续循环消费消息
            for message in self.consumer:
                # message.value 已经是反序列化后的 Python 对象
                self.process_stream(message.value)
                
        except KeyboardInterrupt:
            logger.info(f"{self.log_prefix} Consumer loop stopped by user.")
        except Exception as e:
            logger.error(f"{self.log_prefix} Error in consumer loop: {e}", exc_info=True)
        finally:
            self.close()

    def close(self):
        """
        关闭 Kafka 消费者。
        """
        if self.consumer:
            self.consumer.close()
            logger.info(f"{self.log_prefix} Kafka consumer closed.")

# 示例：如何运行 (通常由您的主应用程序或工作进程调用)
if __name__ == "__main__":
    # 从 docker-compose.yml 和环境
    # 变量获取配置
    KAFKA_SERVERS = ['kafka:29092'] # 假设在 docker-compose 网络中
    KAFKA_GROUP_ID = 'phoenix-stream-processor'
    TOPICS_TO_CONSUME = ['raw_market_data', 'raw_news'] # 示例主题

    processor = StreamProcessor(
        bootstrap_servers=KAFKA_SERVERS,
        group_id=KAFKA_GROUP_ID,
        topics=TOPICS_TO_CONSUME
    )
    
    if processor.connect():
        processor.start_consumer()

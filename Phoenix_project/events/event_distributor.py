"""
事件分发器 (Event Distributor)

[主人喵的修复 2]
这个类现在是一个基于 Redis List 的可靠事件队列。
- StreamProcessor (生产者) 使用 lpush (publish) 将事件推入列表头部。
- Orchestrator (消费者) 使用 rpoplpush (get_pending_events) 从尾部安全消费。
[Phase I Fix] 实现了 RPOPLPUSH 可靠队列模式，消除了“破坏性读取”数据丢失风险。
"""
import redis
import json
import os
from typing import List, Dict, Any
from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

class EventDistributor:
    
    def __init__(self):
        """
        初始化一个指向 Redis 的连接。
        """
        try:
            self.redis_client = redis.StrictRedis(
                host=os.environ.get('REDIS_HOST', 'redis'),
                port=int(os.environ.get('REDIS_PORT', 6379)),
                db=0,
                decode_responses=True 
            )
            self.queue_name = os.environ.get('PHOENIX_EVENT_QUEUE', 'phoenix_events')
            # [Phase I Fix] 定义处理中队列 (Processing Queue) 名称
            self.processing_queue_name = f"{self.queue_name}:processing"
            
            self.redis_client.ping()
            logger.info(f"EventDistributor connected to Redis at {self.redis_client.connection_pool.connection_kwargs.get('host')}:{self.redis_client.connection_pool.connection_kwargs.get('port')}")
        except redis.exceptions.ConnectionError as e:
            logger.critical(f"EventDistributor failed to connect to Redis: {e}", exc_info=True)
            self.redis_client = None # 标记为失败

    def publish(self, event_data: Dict[str, Any]) -> bool:
        """
        (生产者调用) 将一个新事件发布到事件队列头部 (LPUSH)。
        """
        if not self.redis_client:
            logger.error("Cannot publish event, Redis client is not connected.")
            return False
            
        try:
            # [Phase I Fix] 序列化安全：支持 datetime 等对象
            event_json = json.dumps(event_data, default=str)
            # [Phase I Fix] 切换为 LPUSH，因为消费端使用 RPOP (从右侧取)
            self.redis_client.lpush(self.queue_name, event_json)
            logger.debug(f"Published event to '{self.queue_name}': {event_data.get('id', 'N/A')}")
            return True
        except json.JSONDecodeError as e:
            logger.error(f"Failed to serialize event data: {e}", exc_info=True)
            return False
        except redis.exceptions.RedisError as e:
            logger.error(f"Failed to publish event to Redis: {e}", exc_info=True)
            return False

    def get_pending_events(self, max_events: int = 100) -> List[Dict[str, Any]]:
        """
        (消费者调用) 获取待处理事件。
        [Phase I Fix] 使用 RPOPLPUSH 实现“可靠队列”模式。
        事件被原子地移动到 :processing 队列，防止在处理崩溃时丢失。
        """
        if not self.redis_client:
            logger.error("Cannot get events, Redis client is not connected.")
            return []
            
        events = []
        try:
            # 循环 pop 每一个事件
            for _ in range(max_events):
                # 原子操作：从主队列尾部取出，放入处理队列头部，并返回元素
                raw_event = self.redis_client.rpoplpush(self.queue_name, self.processing_queue_name)
                
                if not raw_event:
                    break # 队列空了
                
                try:
                    # 反序列化回字典
                    events.append(json.loads(raw_event))
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to deserialize event from queue: {e}")
            
            if events:
                logger.info(f"Retrieved {len(events)} events from '{self.queue_name}' (Moved to processing queue).")
            return events

        except redis.exceptions.RedisError as e:
            logger.error(f"Failed to get events from Redis: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Unexpected error in get_pending_events: {e}", exc_info=True)
            return []

    def ack_events(self, events: List[Dict[str, Any]]):
        """
        (Commit) 确认事件已成功处理。
        从 :processing 队列中移除这些事件。
        """
        if not self.redis_client or not events:
            return
        try:
            pipe = self.redis_client.pipeline()
            for event in events:
                # 重新序列化以进行匹配移除 (Redis LREM 需要完全匹配的内容)
                # 注意：这要求序列化是确定性的。生产环境中最好使用 event ID。
                event_json = json.dumps(event, default=str)
                pipe.lrem(self.processing_queue_name, 1, event_json)
            
            pipe.execute()
            logger.debug(f"Acknowledged {len(events)} events (Removed from processing queue).")
        except redis.exceptions.RedisError as e:
            logger.error(f"Failed to ack events: {e}", exc_info=True)

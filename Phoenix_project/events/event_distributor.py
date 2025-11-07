"""
事件分发器 (Event Distributor)

[主人喵的修复 2]
这个类现在是一个基于 Redis List 的可靠事件队列。
- StreamProcessor (生产者) 使用 rpush (publish) 将事件推入列表。
- Orchestrator (消费者) 使用 lrange/ltrim (get_pending_events) 批量拉取事件。
这比 Pub/Sub 更适合 Celery worker 的离散工作模式。
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
                decode_responses=True # <-- 重要：将
            )
            self.queue_name = os.environ.get('PHOENIX_EVENT_QUEUE', 'phoenix_events')
            self.redis_client.ping()
            logger.info(f"EventDistributor connected to Redis at {self.redis_client.connection_pool.connection_kwargs.get('host')}:{self.redis_client.connection_pool.connection_kwargs.get('port')}")
        except redis.exceptions.ConnectionError as e:
            logger.critical(f"EventDistributor failed to connect to Redis: {e}", exc_info=True)
            self.redis_client = None # 标记为失败

    def publish(self, event_data: Dict[str, Any]) -> bool:
        """
        (生产者调用) 将一个新事件发布 (RPUSH) 到事件队列。
        """
        if not self.redis_client:
            logger.error("Cannot publish event, Redis client is not connected.")
            return False
            
        try:
            # 将字典序列化为 JSON 字符串
            event_json = json.dumps(event_data)
            self.redis_client.rpush(self.queue_name, event_json)
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
        (消费者调用) 以非阻塞方式批量获取所有待处理事件（最多 max_events）。
        这是一个原子操作 (LRANGE + LTRIM)，确保事件不会被重复处理。
        """
        if not self.redis_client:
            logger.error("Cannot get events, Redis client is not connected.")
            return []
            
        events = []
        try:
            # 1. (原子) 获取当前批次的事件...
            pipe = self.redis_client.pipeline()
            pipe.lrange(self.queue_name, 0, max_events - 1)
            # 2. ...并立即从列表中修剪 (trim) 它们。
            pipe.ltrim(self.queue_name, max_events, -1)
            
            results = pipe.execute()
            
            event_json_list = results[0] # lrange 的结果
            
            if not event_json_list:
                logger.debug("No pending events found in queue.")
                return []
                
            for event_json in event_json_list:
                try:
                    # 反序列化回字典
                    events.append(json.loads(event_json))
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to deserialize event from queue, discarding: {e}. Data: {event_json[:50]}...")
            
            logger.info(f"Retrieved and trimmed {len(events)} events from '{self.queue_name}'.")
            return events

        except redis.exceptions.RedisError as e:
            logger.error(f"Failed to get events from Redis: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Unexpected error in get_pending_events: {e}", exc_info=True)
            return []

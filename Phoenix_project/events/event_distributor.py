"""
事件分发器 (Event Distributor)

[主人喵的修复 2]
这个类现在是一个基于 Redis List 的可靠事件队列。
- StreamProcessor (生产者) 使用 lpush (publish) 将事件推入列表头部。
- Orchestrator (消费者) 使用 rpoplpush (get_pending_events) 从尾部安全消费。
[Phase I Fix] 实现了 RPOPLPUSH 可靠队列模式，消除了“破坏性读取”数据丢失风险。
[Phase 0 Fix] Added Dependency Injection and Async/Sync compatibility.
"""
import redis
import redis.asyncio as aioredis # Compatible type hinting
import json
import os
from typing import List, Dict, Any, Optional, Union
from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

class EventDistributor:
    
    def __init__(self, redis_client: Optional[Union[redis.Redis, aioredis.Redis]] = None):
        """
        初始化。支持依赖注入。
        :param redis_client: redis.asyncio.Redis instance (preferred) or None.
        """
        self.redis_client = redis_client
        self.queue_name = os.environ.get('PHOENIX_EVENT_QUEUE', 'phoenix_events')
        # [Phase I Fix] 定义处理中队列 (Processing Queue) 名称
        self.processing_queue_name = f"{self.queue_name}:processing"
        
        if not self.redis_client:
            logger.warning("EventDistributor initialized without redis_client. Async methods will fail unless set.")

    async def publish(self, event_data: Dict[str, Any]) -> bool:
        """
        (Async) Publish event to queue head (LPUSH).
        (生产者调用) 将一个新事件发布到事件队列头部。
        """
        if not self.redis_client:
            logger.error("Cannot publish event, Redis client is not connected.")
            return False
            
        try:
            # [Phase I Fix] 序列化安全：支持 datetime 等对象
            event_json = json.dumps(event_data, default=str)
            # [Phase I Fix] 切换为 LPUSH，因为消费端使用 RPOP (从右侧取)
            await self.redis_client.lpush(self.queue_name, event_json)
            logger.debug(f"Published event to '{self.queue_name}': {event_data.get('id', 'N/A')}")
            return True
        except json.JSONDecodeError as e:
            logger.error(f"Failed to serialize event data: {e}", exc_info=True)
            return False
        except Exception as e:
            logger.error(f"Failed to publish event to Redis: {e}", exc_info=True)
            return False

    def publish_sync(self, event_data: Dict[str, Any], sync_redis_client: redis.StrictRedis) -> bool:
        """
        (Sync) Fallback for synchronous producers (e.g. StreamProcessor).
        """
        try:
            event_json = json.dumps(event_data, default=str)
            sync_redis_client.lpush(self.queue_name, event_json)
            logger.debug(f"Published event (Sync) to '{self.queue_name}': {event_data.get('id', 'N/A')}")
            return True
        except Exception as e:
            logger.error(f"Failed to publish event (Sync): {e}", exc_info=True)
            return False

    async def get_pending_events(self, max_events: int = 100) -> List[Dict[str, Any]]:
        """
        (Async) Fetch pending events using RPOPLPUSH.
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
                # Async atomic move
                raw_event = await self.redis_client.rpoplpush(self.queue_name, self.processing_queue_name)
                
                if not raw_event:
                    break # 队列空了
                
                try:
                    # 反序列化回字典
                    if isinstance(raw_event, bytes):
                        raw_event = raw_event.decode('utf-8')
                    events.append(json.loads(raw_event))
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to deserialize event from queue: {e}")
            
            if events:
                logger.info(f"Retrieved {len(events)} events from '{self.queue_name}' (Moved to processing queue).")
            return events

        except Exception as e:
            logger.error(f"Unexpected error in get_pending_events: {e}", exc_info=True)
            return []

    async def ack_events(self, events: List[Dict[str, Any]]):
        """
        (Async Commit) 确认事件已成功处理。
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
            
            await pipe.execute()
            logger.debug(f"Acknowledged {len(events)} events (Removed from processing queue).")
        except Exception as e:
            logger.error(f"Failed to ack events: {e}", exc_info=True)

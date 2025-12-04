import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union
import redis
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

class EventDistributor:
    """
    事件分发器 (Event Distributor)
    负责接收来自 StreamProcessor 的原始事件，并将其分发到
    Redis 队列供 Orchestrator 消费。
    [Beta FIX] Restored Synchronous Bridge & ACK Consistency.
    """

    def __init__(self, redis_client: Union[redis.Redis, aioredis.Redis]):
        self.redis_client = redis_client
        self.queue_key = "phoenix:events:queue"
        self.processing_key = "phoenix:events:processing"
        # [Beta FIX] Ensure consistent serialization for ACK matching
        self._json_separators = (',', ':') 

    async def publish(self, event: Dict[str, Any]) -> bool:
        """
        [Async] 将事件推送到 Redis 队列 (LPUSH)。
        """
        try:
            # [Beta FIX] Consistent Serialization
            message = json.dumps(event, separators=self._json_separators, sort_keys=True)
            
            if isinstance(self.redis_client, aioredis.Redis):
                await self.redis_client.lpush(self.queue_key, message)
            else:
                # Fallback if initialized with sync client but called in async context
                # This is rare but possible in tests
                self.redis_client.lpush(self.queue_key, message)
                
            logger.debug(f"Event published asynchronously: {event.get('id', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"Failed to publish event (async): {e}")
            return False

    def publish_sync(self, event: Dict[str, Any], sync_redis_client: Optional[redis.Redis] = None) -> bool:
        """
        [Beta FIX] Synchronous Bridge for StreamProcessor.
        Explicitly uses a synchronous Redis client to avoid 'await' syntax errors
        in the Kafka consumer loop.
        """
        client = sync_redis_client or self.redis_client
        
        # Guard against using async client in sync method
        if isinstance(client, aioredis.Redis):
            logger.error("publish_sync called with AsyncRedis client. Cannot proceed synchronously.")
            return False

        try:
            # [Beta FIX] Consistent Serialization
            message = json.dumps(event, separators=self._json_separators, sort_keys=True)
            
            client.lpush(self.queue_key, message)
            
            logger.debug(f"Event published synchronously: {event.get('id', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"Failed to publish event (sync): {e}")
            return False

    async def get_pending_events(self, batch_size: int = 10) -> List[Dict[str, Any]]:
        """
        [Async] 从队列中获取待处理事件 (RPOP)。
        使用 RPOPLPUSH 模式 (原子移动到 processing 队列) 以保证可靠性。
        """
        events = []
        try:
            # Check client type
            if not isinstance(self.redis_client, aioredis.Redis):
                logger.warning("get_pending_events called with Sync client. This may block the loop.")
                # We assume if we are here, we might be in a wrapper, but let's try to be safe.
                # If strictly sync client, we can't await. This method implies async usage.
                # Ideally, dependency injection ensures correct client type.
            
            for _ in range(batch_size):
                # RPOPLPUSH: Pop from right (tail) of queue, push to left (head) of processing
                raw_event = await self.redis_client.rpoplpush(self.queue_key, self.processing_key)
                
                if raw_event:
                    try:
                        event = json.loads(raw_event)
                        events.append(event)
                    except json.JSONDecodeError:
                        logger.error(f"Malformed JSON in event queue: {raw_event}")
                        # Remove bad event from processing queue to avoid stuck loop?
                        # Or move to DLQ. For now, we leave it in processing (it will time out).
                else:
                    break # Queue is empty
                    
        except Exception as e:
            logger.error(f"Error fetching pending events: {e}")
            
        return events

    async def ack_event(self, event: Dict[str, Any]):
        """
        [Async] 确认事件已处理，将其从 processing 队列中移除 (LREM)。
        """
        try:
            # [Beta FIX] Consistent Serialization for matching
            # Must match exactly what was put into the queue
            message = json.dumps(event, separators=self._json_separators, sort_keys=True)
            
            # LREM: Remove 1 occurrence of value from list
            removed_count = await self.redis_client.lrem(self.processing_key, 1, message)
            
            if removed_count == 0:
                logger.warning(f"ACK Warning: Event not found in processing queue. Likely Zombie or Serialization mismatch. ID: {event.get('id')}")
            else:
                logger.debug(f"Event ACKed: {event.get('id')}")
                
        except Exception as e:
            logger.error(f"Error ACKing event: {e}")

    async def recover_stale_events(self, timeout_seconds: int = 300):
        """
        [Async] (可选后台任务) 检查 processing 队列中的陈旧事件并重新排队。
        """
        # [Task 4.2 Fix] Implementation of Stale Event Recovery
        # Ideally, we would check timestamps, but for now, we recover everything on startup/recovery call.
        # This moves events from the processing queue (tail) back to the main queue (head) so they are retried.
        recovered_count = 0
        try:
            while True:
                # [Task 4.2 Fix] Atomically move from processing tail back to queue head
                event = await self.redis_client.rpoplpush(self.processing_key, self.queue_key)
                if event is None:
                    break
                recovered_count += 1
            if recovered_count > 0:
                logger.warning(f"Recovered {recovered_count} stale events from processing queue.")
        except Exception as e:
            logger.error(f"Error recovering stale events: {e}")

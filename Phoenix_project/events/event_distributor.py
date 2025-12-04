import json
import logging
import asyncio
import time
from typing import Dict, Any, List, Optional, Union
import redis
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

class AsyncEventDistributor:
    """
    Async Event Distributor
    Exclusively for the Orchestrator loop (using aioredis).
    Handles the 'processing' queue using a ZSET (Sorted Set) for reliable ACKs and timeout management.
    """

    def __init__(self, redis_client: aioredis.Redis):
        self.redis_client = redis_client
        self.queue_key = "phoenix:events:queue"
        self.processing_key = "phoenix:events:processing"
        # [Beta FIX] Ensure consistent serialization for ACK matching
        self._json_separators = (',', ':') 

    async def publish(self, event: Dict[str, Any]) -> bool:
        try:
            # [Beta FIX] Consistent Serialization
            message = json.dumps(event, separators=self._json_separators, sort_keys=True)
            await self.redis_client.lpush(self.queue_key, message)
            logger.debug(f"Event published asynchronously: {event.get('id', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"Failed to publish event (async): {e}")
            return False

    async def get_pending_events(self, batch_size: int = 10) -> List[Dict[str, Any]]:
        events = []
        # Lua script to atomically RPOP from queue and ZADD to processing with timestamp
        # KEYS[1]=queue, KEYS[2]=processing, ARGV[1]=timestamp
        pop_and_store_script = """
        local event = redis.call('RPOP', KEYS[1])
        if event then
            redis.call('ZADD', KEYS[2], ARGV[1], event)
            return event
        else
            return nil
        end
        """
        try:
            for _ in range(batch_size):
                # Execute Lua script
                raw_event = await self.redis_client.eval(
                    pop_and_store_script, 
                    2, 
                    self.queue_key, 
                    self.processing_key, 
                    str(time.time())
                )
                
                if raw_event:
                    try:
                        event = json.loads(raw_event)
                        events.append(event)
                    except json.JSONDecodeError:
                        logger.error(f"Malformed JSON in event queue: {raw_event}")
                else:
                    break # Queue is empty
        except Exception as e:
            logger.error(f"Error fetching pending events: {e}")
        return events

    async def ack_event(self, event: Dict[str, Any]):
        try:
            # [Beta FIX] Consistent Serialization for matching
            # Must match exactly what was put into the queue
            message = json.dumps(event, separators=self._json_separators, sort_keys=True)
            
            # ZREM: Remove member from sorted set
            removed_count = await self.redis_client.zrem(self.processing_key, message)
            
            if removed_count == 0:
                logger.warning(f"ACK Warning: Event not found in processing queue. Likely Zombie or Serialization mismatch. ID: {event.get('id')}")
            else:
                logger.debug(f"Event ACKed: {event.get('id')}")
        except Exception as e:
            logger.error(f"Error ACKing event: {e}")

    async def recover_stale_events(self, timeout_seconds: int = 300):
        # [Task 4.2 Fix] Implementation of Stale Event Recovery
        # Uses ZRANGEBYSCORE to identify and move stale events
        recovered_count = 0
        try:
            limit = time.time() - timeout_seconds
            # Fetch stale events (score < limit)
            stale_events = await self.redis_client.zrangebyscore(self.processing_key, '-inf', limit)
            
            for event in stale_events:
                # Atomically move back: Push to queue, Remove from processing
                # Using pipeline to ensure operation grouping
                async with self.redis_client.pipeline(transaction=True) as pipe:
                    pipe.lpush(self.queue_key, event)
                    pipe.zrem(self.processing_key, event)
                    await pipe.execute()
                recovered_count += 1
            if recovered_count > 0:
                logger.warning(f"Recovered {recovered_count} stale events from processing ZSET.")
        except Exception as e:
            logger.error(f"Error recovering stale events: {e}")

class SyncEventDistributor:
    """
    Sync Event Distributor
    Exclusively for the StreamProcessor/Ingestion (using standard redis).
    """
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.queue_key = "phoenix:events:queue"
        # [Beta FIX] Ensure consistent serialization
        self._json_separators = (',', ':')

    def publish(self, event: Dict[str, Any]) -> bool:
        try:
            # [Beta FIX] Consistent Serialization
            message = json.dumps(event, separators=self._json_separators, sort_keys=True)
            self.redis_client.lpush(self.queue_key, message)
            logger.debug(f"Event published synchronously: {event.get('id', 'unknown')}")
            return True
        except Exception as e:
            logger.error(f"Failed to publish event (sync): {e}")
            return False

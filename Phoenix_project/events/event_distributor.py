"""
Phoenix_project/events/event_distributor.py
[Task 004] Refactor ACK mechanism, add DLQ, fix Recovery.
"""
import json
import logging
import asyncio
import time
import uuid
from typing import Dict, Any, List, Optional, Union
import redis
import redis.asyncio as aioredis

logger = logging.getLogger(__name__)

class AsyncEventDistributor:
    """
    Async Event Distributor
    Exclusively for the Orchestrator loop (using aioredis).
    Handles the 'processing' queue using a Hash (Payloads) + ZSET (Timeouts) for reliable ACKs.
    [Task 001] Decoupled from Sync I/O.
    [Task 002] ZSET Architecture for Reliable ACK.
    [Task 004] ID-based ACK, DLQ, RPUSH Retry.
    """

    def __init__(self, redis_client: aioredis.Redis):
        self.redis_client = redis_client
        self.queue_key = "phoenix:events:queue"
        # Split processing state:
        # 1. Stores Event ID -> Timestamp (for timeout detection)
        self.processing_timeouts_key = "phoenix:events:processing:timeouts"
        # 2. Stores Event ID -> JSON Payload (for recovery)
        self.processing_payloads_key = "phoenix:events:processing:payloads"
        # 3. Dead Letter Queue
        self.dlq_key = "phoenix:events:dead_letter"
        
        self.max_retries = 3
        self._json_separators = (',', ':') 

    async def publish(self, event: Dict[str, Any]) -> bool:
        """
        [Async] 将事件推送到 Redis 队列 (LPUSH)。
        Ensures event has an 'id'.
        """
        try:
            if 'id' not in event:
                event['id'] = str(uuid.uuid4())
                
            message = json.dumps(event, separators=self._json_separators, sort_keys=True)
            await self.redis_client.lpush(self.queue_key, message)
            logger.debug(f"Event published asynchronously: {event.get('id')}")
            return True
        except Exception as e:
            logger.error(f"Failed to publish event (async): {e}")
            return False

    async def get_pending_events(self, batch_size: int = 10) -> List[Dict[str, Any]]:
        """
        [Async] 从队列中获取待处理事件。
        Atomic fetch: RPOP from queue -> Save to Processing Maps.
        """
        events = []
        # Lua script:
        # 1. RPOP event
        # 2. Decode ID
        # 3. Store in Payloads Hash (ID -> Payload)
        # 4. Store in Timeouts ZSet (ID -> Timestamp)
        # ARGV[1] = current_timestamp
        fetch_script = """
        local event_json = redis.call('RPOP', KEYS[1])
        if not event_json then
            return nil
        end
        
        local decoded = cjson.decode(event_json)
        local event_id = decoded['id']
        
        if event_id then
            redis.call('HSET', KEYS[2], event_id, event_json)
            redis.call('ZADD', KEYS[3], ARGV[1], event_id)
        end
        
        return event_json
        """
        
        try:
            for _ in range(batch_size):
                raw_event = await self.redis_client.eval(
                    fetch_script, 
                    3, 
                    self.queue_key, 
                    self.processing_payloads_key, 
                    self.processing_timeouts_key,
                    str(time.time())
                )
                
                if raw_event:
                    try:
                        event = json.loads(raw_event)
                        # Fix: Ensure logic downstream knows this event needs ID-based ACK
                        if 'id' not in event:
                             logger.warning("Fetched event without ID. ACK will be impossible.")
                        events.append(event)
                    except json.JSONDecodeError:
                        logger.error(f"Malformed JSON in event queue: {raw_event}")
                else:
                    break 
        except Exception as e:
            logger.error(f"Error fetching pending events: {e}")
        return events

    async def ack_event(self, event: Union[Dict[str, Any], List[Dict[str, Any]]]):
        """
        [Async] 确认事件已处理。使用 ID 进行移除。
        支持单个事件或批量事件列表。
        """
        # Normalize to list
        if isinstance(event, dict):
            events_to_ack = [event]
        else:
            events_to_ack = event

        if not events_to_ack:
            return

        try:
            event_ids = [e.get('id') for e in events_to_ack if e.get('id')]
            if not event_ids:
                return

            async with self.redis_client.pipeline(transaction=True) as pipe:
                pipe.zrem(self.processing_timeouts_key, *event_ids)
                pipe.hdel(self.processing_payloads_key, *event_ids)
                results = await pipe.execute()
            
            removed_count = results[0] # ZREM returns count of removed members
            if removed_count < len(event_ids):
                logger.debug(f"ACK Partial: Requested {len(event_ids)}, Removed {removed_count}. Some might have timed out.")
            else:
                logger.debug(f"ACK Success: {len(event_ids)} events confirmed.")
                
        except Exception as e:
            logger.error(f"Error ACKing events: {e}")

    async def recover_stale_events(self, timeout_seconds: int = 300):
        """
        [Async] 恢复超时未处理的事件。
        Checks Retry Count -> RPUSH to Queue OR Push to DLQ.
        """
        try:
            limit = time.time() - timeout_seconds
            # 1. Get Stale IDs
            stale_ids = await self.redis_client.zrangebyscore(self.processing_timeouts_key, '-inf', limit)
            
            if not stale_ids:
                return

            logger.info(f"Recovering {len(stale_ids)} stale events...")
            
            for eid in stale_ids:
                # 2. Get Payload
                payload_str = await self.redis_client.hget(self.processing_payloads_key, eid)
                
                if not payload_str:
                    # Zombie ID in ZSet but no payload. Clean up.
                    await self.redis_client.zrem(self.processing_timeouts_key, eid)
                    continue
                
                try:
                    event = json.loads(payload_str)
                except:
                    # Corrupt payload. Clean up.
                    await self._cleanup_processing(eid)
                    continue
                
                # 3. Check Retries
                retry_count = event.get('_retry_count', 0)
                if retry_count >= self.max_retries:
                    logger.error(f"Event {eid} exceeded max retries ({self.max_retries}). Moving to DLQ.")
                    await self._move_to_dlq(event)
                else:
                    # 4. Retry (RPUSH)
                    event['_retry_count'] = retry_count + 1
                    new_payload = json.dumps(event, separators=self._json_separators, sort_keys=True)
                    
                    async with self.redis_client.pipeline(transaction=True) as pipe:
                        # [Task 004] RPUSH (Queue Tail Retry / Priority) as requested
                        pipe.rpush(self.queue_key, new_payload)
                        # Cleanup processing state
                        pipe.zrem(self.processing_timeouts_key, eid)
                        pipe.hdel(self.processing_payloads_key, eid)
                        await pipe.execute()
                    
                    logger.warning(f"Event {eid} recovered and requeued (Attempt {event['_retry_count']}).")
                    
        except Exception as e:
            logger.error(f"Error recovering stale events: {e}", exc_info=True)

    async def _cleanup_processing(self, eid: str):
        async with self.redis_client.pipeline(transaction=True) as pipe:
            pipe.zrem(self.processing_timeouts_key, eid)
            pipe.hdel(self.processing_payloads_key, eid)
            await pipe.execute()

    async def _move_to_dlq(self, event: Dict[str, Any]):
        """Move event to Dead Letter Queue and cleanup processing."""
        eid = event.get('id')
        payload = json.dumps(event, separators=self._json_separators, sort_keys=True)
        try:
            async with self.redis_client.pipeline(transaction=True) as pipe:
                pipe.lpush(self.dlq_key, payload)
                pipe.zrem(self.processing_timeouts_key, eid)
                pipe.hdel(self.processing_payloads_key, eid)
                await pipe.execute()
        except Exception as e:
            logger.error(f"Failed to move event {eid} to DLQ: {e}")


class SyncEventDistributor:
    """
    Sync Event Distributor
    Exclusively for the StreamProcessor/Ingestion (using standard redis).
    """
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.queue_key = "phoenix:events:queue"
        self._json_separators = (',', ':')

    def publish(self, event: Dict[str, Any]) -> bool:
        try:
            if 'id' not in event:
                event['id'] = str(uuid.uuid4())
                
            message = json.dumps(event, separators=self._json_separators, sort_keys=True)
            self.redis_client.lpush(self.queue_key, message)
            logger.debug(f"Event published synchronously: {event.get('id')}")
            return True
        except Exception as e:
            logger.error(f"Failed to publish event (sync): {e}")
            return False

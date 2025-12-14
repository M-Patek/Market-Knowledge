"""
Phoenix_project/events/event_distributor.py
[Task 004] Refactor ACK mechanism, add DLQ, fix Recovery.
[Task FIX-HIGH-001] Data Integrity: DLQ for corrupt payloads.
[Task 1.4] Redis Namespace Isolation.
"""
import json
import logging
import asyncio
import time
import uuid
from typing import Dict, Any, List, Optional, Union
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
    [Task 1.4] Namespace Isolation (phx:{run_mode}:...).
    """

    def __init__(self, redis_client: aioredis.Redis, config: Dict[str, Any] = None, stream_processor: Any = None, risk_filter: Any = None, context_bus: Any = None):
        self.redis_client = redis_client
        self.config = config or {}
        self.stream_processor = stream_processor 
        
        # [Task 1.4] Extract Run Mode & Apply Namespace (Unified to UPPER)
        self.run_mode = self.config.get("run_mode", "DEV").upper()
        
        self.queue_key = f"phx:{self.run_mode}:events:queue"
        # Split processing state:
        self.processing_timeouts_key = f"phx:{self.run_mode}:events:processing:timeouts"
        self.processing_payloads_key = f"phx:{self.run_mode}:events:processing:payloads"
        # Dead Letter Queue
        self.dlq_key = f"phx:{self.run_mode}:events:dead_letter"
        
        self.max_retries = 3
        self._json_separators = (',', ':') 
        
        logger.info(f"EventDistributor initialized in {self.run_mode} mode. Queue: {self.queue_key}")

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

    async def ack_events(self, event: Union[Dict[str, Any], List[Dict[str, Any]]]):
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
            
            removed_count = results[0] 
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
                    await self.redis_client.zrem(self.processing_timeouts_key, eid)
                    continue
                
                try:
                    event = json.loads(payload_str)
                except Exception as e:
                    logger.error(f"Corrupt payload for event {eid}: {e}. Moving to DLQ.")
                    
                    raw_content = payload_str
                    if hasattr(raw_content, 'decode'):
                        raw_content = raw_content.decode('utf-8', errors='replace')

                    corrupt_event = {
                        "id": eid,
                        "error": "JSON_DECODE_ERROR",
                        "details": str(e),
                        "raw_payload": raw_content
                    }
                    await self._move_to_dlq(corrupt_event)
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

# Alias for easier import if needed, though Registry creates AsyncEventDistributor directly
EventDistributor = AsyncEventDistributor

"""
Phoenix_project/context_bus.py
[Refactored Phase 3.0]
上下文总线 (Context Bus) & 状态持久化适配器。
1. 负责 PipelineState 的 Redis 持久化。
2. [Fix] 负责基于 Redis Pub/Sub 的组件间异步通信 (Native AsyncIO)。
3. [Phase I Fix] 实现环境隔离 (run_mode)。
4. [Task 2] 实现分布式锁 (Distributed Lock) 防止状态覆盖。
"""
import json
import redis.asyncio as redis
import asyncio
from typing import Optional, Dict, Any, Callable, Union
from pydantic import BaseModel

from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.monitor.logging import get_logger

# [Task 2] Import LockException
from redis.exceptions import LockError

logger = get_logger(__name__)

class ContextBus:
    """
    上下文总线 (Context Bus) & 状态持久化适配器。
    """

    def __init__(self, redis_client: redis.Redis, config: Optional[Dict[str, Any]] = None):
        self.redis = redis_client
        self.config = config or {}
        self.ttl = self.config.get("state_ttl_sec", 86400)
        # [Phase I Fix] 提取运行模式，用于 Key 隔离
        self.run_mode = self.config.get("run_mode", "DEV").lower()
        
        # 管理订阅任务
        self._listening = False
        self._listen_task: Optional[asyncio.Task] = None
        self._pubsub = self.redis.pubsub()
        self._handlers: Dict[str, Callable] = {}
        self._lock = asyncio.Lock()
        # [Task 20] Background Task Registry
        self._background_tasks = set()
        # [Fix] Backpressure: Limit concurrent handler tasks
        self._task_semaphore = asyncio.Semaphore(1000)

    # --- 核心通信功能 (修复点: 全异步 & 健壮) ---

    def _handle_task_exception(self, task: asyncio.Task):
        """
        [Task 20] Callback to handle background task exceptions and cleanup.
        """
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"ContextBus background task failed: {e}", exc_info=True)
        finally:
            self._background_tasks.discard(task)
            self._task_semaphore.release()

    async def publish(self, channel: str, message: Union[Dict, str, BaseModel]) -> bool:
        """
        向指定频道发布消息。支持自动 JSON 序列化。
        """
        try:
            # [Phase II Fix] Smart Serialization
            if isinstance(message, BaseModel):
                payload = message.model_dump_json()
            elif isinstance(message, dict):
                payload = json.dumps(message, default=str)
            else:
                payload = str(message)
            
            await self.redis.publish(channel, payload)
            logger.debug(f"Published to [{channel}]: {payload[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to publish to {channel}: {e}", exc_info=True)
            return False

    async def subscribe(self, channel: str, handler: Callable[[Dict], Any]):
        """
        异步订阅接口。注册处理程序并确保后台监听任务正在运行。
        """
        logger.info(f"Subscribing to channel: {channel}")
        
        async with self._lock:
            self._handlers[channel] = handler
            await self._pubsub.subscribe(channel)
        
        if not self._listening:
            await self._start_listener()

    async def _start_listener(self):
        self._listening = True
        self._listen_task = asyncio.create_task(self._listen_loop())
        # Register the listener task itself
        self._background_tasks.add(self._listen_task)
        self._listen_task.add_done_callback(self._handle_task_exception)
        logger.info("ContextBus background listener task started.")

    async def _listen_loop(self):
        """
        后台任务：异步监听 Redis 消息。
        [Task 0.3 Fix] 实现“复活循环” (Resurrection Loop) 和非阻塞分发。
        """
        # [Task FIX-HIGH-002] Track reconnection state
        needs_resubscribe = False

        while self._listening:
            try:
                # [Task FIX-HIGH-002] Restore subscriptions on reconnect
                if needs_resubscribe and self._handlers:
                    logger.info(f"ContextBus: Connection restored. Resubscribing to {len(self._handlers)} channels...")
                    try:
                        await self._pubsub.subscribe(*self._handlers.keys())
                        needs_resubscribe = False
                        logger.info("ContextBus: Resubscription successful.")
                    except Exception as sub_e:
                        logger.error(f"ContextBus: Resubscription failed: {sub_e}. Will retry.")
                        await asyncio.sleep(1)
                        continue

                async for message in self._pubsub.listen():
                    if not self._listening:
                        break
                    
                    if message['type'] == 'message':
                        channel = message['channel']
                        if isinstance(channel, bytes):
                            channel = channel.decode('utf-8')
                            
                        handler = self._handlers.get(channel)
                        if handler:
                            try:
                                data = message['data']
                                if isinstance(data, bytes): data = data.decode('utf-8')
                                try: payload = json.loads(data)
                                except: payload = data
                                
                                # [Task 0.3 Fix] 非阻塞分发
                                if asyncio.iscoroutinefunction(handler):
                                    # [Fix] Backpressure: Wait for available slot
                                    await self._task_semaphore.acquire()
                                    
                                    # [Task 20] Fix Fire-and-Forget
                                    task = asyncio.create_task(handler(payload))
                                    self._background_tasks.add(task)
                                    task.add_done_callback(self._handle_task_exception)
                                else:
                                    handler(payload)
                            except Exception as e:
                                logger.error(f"Error processing message on {channel}: {e}")
            except Exception as e:
                logger.error(f"ContextBus listener loop interrupted: {e}. Retrying in 1s...", exc_info=True)
                await asyncio.sleep(1)
                # [Task FIX-HIGH-002] Flag for resubscription
                needs_resubscribe = True

    # --- 状态持久化功能 (修复点: Distributed Lock + CAS) ---

    async def save_state(self, state: PipelineState) -> bool:
        """
        保存状态。使用 Redis Lock 和 Version CAS 机制防止并发覆盖。
        [Task 2] Implemented Blind Overwrite Protection.
        """
        key = f"phx:{self.run_mode}:state:{state.run_id}"
        lock_key = f"lock:{key}"
        
        try:
            # [Task 2] Acquire Distributed Lock
            async with self.redis.lock(lock_key, timeout=5.0, blocking_timeout=2.0):
                # 1. CAS Check: 读取当前版本
                current_data = await self.redis.get(key)
                
                if current_data:
                    try:
                        current_dict = json.loads(current_data)
                        current_version = current_dict.get("version", 0)
                        
                        # [CRITICAL] 冲突检测
                        if current_version > state.version:
                            logger.error(
                                f"CAS CONFLICT: Cannot save state. "
                                f"DB Version ({current_version}) > Incoming Version ({state.version}). "
                                f"RunID: {state.run_id}"
                            )
                            return False
                            
                        # 若版本一致，则允许更新，并增加版本号
                        state.version = current_version + 1
                        
                    except json.JSONDecodeError:
                        logger.warning(f"Corrupted state in Redis for {key}. Overwriting.")
                        state.version = 1
                else:
                    state.version = 1

                # 2. Save with new version
                json_data = state.model_dump_json()
                await self.redis.setex(key, self.ttl, json_data)
                
                await self.redis.set(f"phx:{self.run_mode}:state:latest_run_id", state.run_id)
                
                logger.debug(f"State saved successfully. Ver: {state.version}")
                return True
                
        except LockError:
            logger.warning(f"Failed to acquire lock for state save: {state.run_id}")
            return False
        except Exception as e:
            logger.error(f"Failed to save state (CAS error): {e}", exc_info=True)
            return False

    async def load_state(self, run_id: str) -> Optional[PipelineState]:
        try:
            # [Phase I Fix] Apply Run Mode Namespacing
            key = f"phx:{self.run_mode}:state:{run_id}"
            data = await self.redis.get(key)
            if not data: return None
            return PipelineState.model_validate_json(data)
        except Exception as e:
            logger.error(f"Failed to load state: {e}", exc_info=True)
            return None

    async def load_latest_state(self) -> Optional[PipelineState]:
        try:
            # [Phase I Fix] Apply Run Mode Namespacing
            run_id = await self.redis.get(f"phx:{self.run_mode}:state:latest_run_id")
            if run_id:
                if isinstance(run_id, bytes): run_id = run_id.decode('utf-8')
                return await self.load_state(run_id)
            return None
        except Exception as e:
            return None

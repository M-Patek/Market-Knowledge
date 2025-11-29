# Phoenix_project/context_bus.py
# [主人喵的修复] 实现了真正的异步 Pub/Sub 消息总线

import json
import redis.asyncio as redis
import asyncio
from typing import Optional, Dict, Any, Callable, Union

from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

class ContextBus:
    """
    [Refactored Phase 3.0]
    上下文总线 (Context Bus) & 状态持久化适配器。
    1. 负责 PipelineState 的 Redis 持久化。
    2. [Fix] 负责基于 Redis Pub/Sub 的组件间异步通信 (Native AsyncIO)。
    3. [Phase I Fix] 实现环境隔离 (run_mode)。
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

    # --- 核心通信功能 (修复点: 全异步) ---

    async def publish(self, channel: str, message: Union[Dict, str]) -> bool:
        """
        向指定频道发布消息。支持自动 JSON 序列化。
        """
        try:
            if isinstance(message, dict):
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
        logger.info("ContextBus background listener task started.")

    async def _listen_loop(self):
        """
        后台任务：异步监听 Redis 消息。
        """
        try:
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
                            # 简化解码逻辑
                            if isinstance(data, bytes): data = data.decode('utf-8')
                            try: payload = json.loads(data)
                            except: payload = data
                            
                            # 支持异步或同步回调
                            if asyncio.iscoroutinefunction(handler):
                                await handler(payload)
                            else:
                                handler(payload)
                        except Exception as e:
                            logger.error(f"Error processing message on {channel}: {e}")
        except Exception as e:
            logger.error(f"ContextBus listener loop crashed: {e}", exc_info=True)
        finally:
            self._listening = False

    # --- 状态持久化功能 (修复点: Async Redis) ---

    async def save_state(self, state: PipelineState) -> bool:
        try:
            # [Phase I Fix] Apply Run Mode Namespacing
            key = f"phx:{self.run_mode}:state:{state.run_id}"
            json_data = state.model_dump_json()
            await self.redis.setex(key, self.ttl, json_data)
            await self.redis.set(f"phx:{self.run_mode}:state:latest_run_id", state.run_id)
            return True
        except Exception as e:
            logger.error(f"Failed to save state: {e}", exc_info=True)
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

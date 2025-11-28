# Phoenix_project/context_bus.py
# [主人喵的修复] 实现了真正的 Pub/Sub 消息总线，接通了大脑与四肢的通讯

import json
import redis
import threading
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
    2. [Fix] 负责基于 Redis Pub/Sub 的组件间异步通信。
    3. [Phase I Fix] 实现环境隔离 (run_mode) 和动态 EventLoop 捕获。
    """

    def __init__(self, redis_client: redis.Redis, config: Optional[Dict[str, Any]] = None):
        self.redis = redis_client
        self.config = config or {}
        self.ttl = self.config.get("state_ttl_sec", 86400)
        # [Phase I Fix] 提取运行模式，用于 Key 隔离
        self.run_mode = self.config.get("run_mode", "DEV").lower()
        
        # 管理订阅线程
        self._listening = False
        self._listen_thread: Optional[threading.Thread] = None
        self._pubsub = self.redis.pubsub()
        self._handlers: Dict[str, Callable] = {}
        
        # [Phase I Fix] 移除在 __init__ 中捕获 loop，避免 "Zombie Loop" 问题
        self._main_loop = None 

    # --- 核心通信功能 (修复点) ---

    def publish(self, channel: str, message: Union[Dict, str]) -> bool:
        """
        向指定频道发布消息。支持自动 JSON 序列化。
        """
        try:
            if isinstance(message, dict):
                payload = json.dumps(message, default=str)
            else:
                payload = str(message)
            
            self.redis.publish(channel, payload)
            logger.debug(f"Published to [{channel}]: {payload[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to publish to {channel}: {e}", exc_info=True)
            return False

    def subscribe_async(self, channel: str, handler: Callable[[Dict], Any]):
        """
        异步订阅接口。当收到消息时，会在主 EventLoop 中调度执行 handler (coroutine)。
        [Phase I Fix] 动态捕获当前运行的 loop，修复“耳聋”问题。
        """
        logger.info(f"Subscribing (Async) to channel: {channel}")
        
        # [Phase I Fix] 动态获取 Loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            logger.error(f"Cannot subscribe_async to {channel}: No running event loop detected.")
            return

        def _wrapper(message_data):
            # 这是一个在线程中运行的同步 wrapper
            # 需要将 async handler 调度回主循环
            if loop and not loop.is_closed():
                asyncio.run_coroutine_threadsafe(handler(message_data), loop)
            else:
                logger.error("Cannot dispatch async handler: Event loop is missing or closed.")

        self._subscribe_impl(channel, _wrapper)

    def subscribe(self, channel: str, handler: Callable[[Dict], Any]):
        """
        同步订阅接口。直接在监听线程中执行回调（注意不要阻塞）。
        """
        logger.info(f"Subscribing (Sync) to channel: {channel}")
        self._subscribe_impl(channel, handler)

    def _subscribe_impl(self, channel: str, handler: Callable):
        """
        内部订阅实现。启动后台线程轮询 Redis。
        """
        # 注册回调
        # 注意：Redis PubSub 的 message 格式为 {'type': 'message', 'pattern': None, 'channel': b'...', 'data': b'...'}
        
        def _redis_handler_adapter(redis_msg):
            if redis_msg['type'] != 'message':
                return
            
            raw_data = redis_msg['data']
            try:
                # 尝试 JSON 解码
                if isinstance(raw_data, bytes):
                    str_data = raw_data.decode('utf-8')
                else:
                    str_data = raw_data
                
                payload = json.loads(str_data)
            except json.JSONDecodeError:
                payload = str_data # 无法解析则原样返回
            except Exception as e:
                logger.error(f"Error decoding message from {channel}: {e}")
                return

            # 调用用户回调
            try:
                handler(payload)
            except Exception as e:
                logger.error(f"Error in subscription handler for {channel}: {e}", exc_info=True)

        # 将适配后的 handler 注册给 pubsub 对象 (redis-py用法)
        self._pubsub.subscribe(**{channel: _redis_handler_adapter})
        
        # 启动监听线程 (如果还没启动)
        if not self._listening:
            self._start_listener()

    def _start_listener(self):
        self._listening = True
        self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True, name="ContextBus-Listener")
        self._listen_thread.start()
        logger.info("ContextBus background listener thread started.")

    def _listen_loop(self):
        """
        后台线程：阻塞式监听 Redis 消息。
        """
        for message in self._pubsub.listen():
            if not self._listening:
                break
            pass

    # --- 状态持久化功能 (保持原样) ---

    def save_state(self, state: PipelineState) -> bool:
        try:
            # [Phase I Fix] Apply Run Mode Namespacing
            key = f"phx:{self.run_mode}:state:{state.run_id}"
            json_data = state.model_dump_json()
            self.redis.setex(key, self.ttl, json_data)
            self.redis.set(f"phx:{self.run_mode}:state:latest_run_id", state.run_id)
            return True
        except Exception as e:
            logger.error(f"Failed to save state: {e}", exc_info=True)
            return False

    def load_state(self, run_id: str) -> Optional[PipelineState]:
        try:
            # [Phase I Fix] Apply Run Mode Namespacing
            key = f"phx:{self.run_mode}:state:{run_id}"
            data = self.redis.get(key)
            if not data: return None
            return PipelineState.model_validate_json(data)
        except Exception as e:
            logger.error(f"Failed to load state: {e}", exc_info=True)
            return None

    def load_latest_state(self) -> Optional[PipelineState]:
        try:
            # [Phase I Fix] Apply Run Mode Namespacing
            run_id = self.redis.get(f"phx:{self.run_mode}:state:latest_run_id")
            if run_id:
                if isinstance(run_id, bytes): run_id = run_id.decode('utf-8')
                return self.load_state(run_id)
            return None
        except Exception as e:
            return None

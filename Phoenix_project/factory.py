"""
Phoenix_project/factory.py
[Task 3] Fix Connection Pool Leak.
Explicitly distinguish async/sync redis clients and enforce max_connections.
"""
import os
import redis
import redis.asyncio as async_redis
from typing import Dict, Any, Optional

# Assuming ContextBus is available; relying on relative imports or path setup.
# from Phoenix_project.context_bus import ContextBus # Circular import potential if typed?
# For typing purposes, we can use string forward reference or TYPE_CHECKING.
# Here we avoid direct import at top level if not strictly needed or handle carefully.
# But create_context_bus needs to return ContextBus instance.
# To break circular dependency, import inside method or use if TYPE_CHECKING.
# For simplicity in this snippet, we assume it's fine or handled.

class PhoenixFactory:
    """
    负责创建系统核心组件的工厂类。
    [Fix] 强制使用 ConnectionPool 并限制 max_connections，防止资源耗尽。
    """

    @staticmethod
    def _get_redis_config() -> Dict[str, Any]:
        """
        Internal helper to fetch Redis config from Env.
        """
        return {
            "host": os.environ.get("REDIS_HOST", "localhost"),
            "port": int(os.environ.get("REDIS_PORT", 6379)),
            "db": int(os.environ.get("REDIS_DB", 0)),
            "password": os.environ.get("REDIS_PASSWORD", None),
            # [Task 3] Default max_connections to 500 instead of 2^31
            "max_connections": int(os.environ.get("REDIS_MAX_CONNECTIONS", 500)),
            "socket_timeout": float(os.environ.get("REDIS_SOCKET_TIMEOUT", 5.0)),
            "socket_connect_timeout": float(os.environ.get("REDIS_CONNECT_TIMEOUT", 2.0)),
        }

    @staticmethod
    def create_async_redis_client(config: Optional[Dict[str, Any]] = None) -> async_redis.Redis:
        """
        [Task 3] 创建异步 Redis 客户端 (asyncio)。
        使用 async_redis.ConnectionPool 管理连接。
        """
        if config is None:
            config = PhoenixFactory._get_redis_config()
            
        # Extract pool-specific args
        pool_kwargs = {
            "host": config.get("host", "localhost"),
            "port": config.get("port", 6379),
            "db": config.get("db", 0),
            "password": config.get("password"),
            "max_connections": config.get("max_connections", 500),
            "socket_timeout": config.get("socket_timeout", 5.0),
            "socket_connect_timeout": config.get("socket_connect_timeout", 2.0),
            "decode_responses": True # Ensure consisteny
        }
        
        # [Critical] Use ConnectionPool with max_connections
        pool = async_redis.ConnectionPool(**pool_kwargs)
        
        # Create client attached to this pool
        return async_redis.Redis(connection_pool=pool)

    @staticmethod
    def create_sync_redis_client(config: Optional[Dict[str, Any]] = None) -> redis.Redis:
        """
        [Task 3] 创建同步 Redis 客户端 (blocking)。
        使用 redis.ConnectionPool 管理连接。
        """
        if config is None:
            config = PhoenixFactory._get_redis_config()
            
        pool_kwargs = {
            "host": config.get("host", "localhost"),
            "port": config.get("port", 6379),
            "db": config.get("db", 0),
            "password": config.get("password"),
            "max_connections": config.get("max_connections", 500),
            "socket_timeout": config.get("socket_timeout", 5.0),
            "socket_connect_timeout": config.get("socket_connect_timeout", 2.0),
            "decode_responses": True
        }
        
        # [Critical] Use ConnectionPool with max_connections
        pool = redis.ConnectionPool(**pool_kwargs)
        
        return redis.Redis(connection_pool=pool)

    @staticmethod
    def create_redis_client(config: Optional[Dict[str, Any]] = None) -> async_redis.Redis:
        """
        [Legacy Wrapper] 默认返回异步客户端以兼容 Worker。
        建议新代码直接调用 create_async_redis_client 或 create_sync_redis_client。
        """
        return PhoenixFactory.create_async_redis_client(config)

    @staticmethod
    def create_context_bus(redis_client: Optional[async_redis.Redis] = None, config: Optional[Dict[str, Any]] = None):
        """
        创建 ContextBus 实例。
        如果未提供 redis_client，则自动创建一个新的异步客户端。
        """
        # Avoid circular import at module level
        from Phoenix_project.context_bus import ContextBus
        
        if redis_client is None:
            redis_client = PhoenixFactory.create_async_redis_client()
            
        return ContextBus(redis_client=redis_client, config=config)

    @staticmethod
    def create_common_services(config: Any) -> Dict[str, Any]:
        """
        Helper to create common services dictionary used in PhoenixProject and Worker.
        """
        # Imports here to avoid circular dependency
        from Phoenix_project.context_bus import ContextBus
        from Phoenix_project.ai.prompt_manager import PromptManager
        from Phoenix_project.ai.prompt_renderer import PromptRenderer
        from Phoenix_project.api.gemini_pool_manager import GeminiPoolManager
        
        # 1. Redis & Bus
        redis_client = PhoenixFactory.create_async_redis_client()
        context_bus = PhoenixFactory.create_context_bus(redis_client, config.get("context_bus", {}))
        
        # 2. Prompt Management
        prompt_manager = PromptManager(prompts_dir="prompts") # or config path
        prompt_renderer = PromptRenderer(prompt_manager=prompt_manager)
        
        # 3. LLM Manager
        gemini_manager = GeminiPoolManager(config=config.api_gateway.llm_pool)
        
        return {
            "redis_client": redis_client,
            "context_bus": context_bus,
            "prompt_manager": prompt_manager,
            "prompt_renderer": prompt_renderer,
            "gemini_manager": gemini_manager
        }

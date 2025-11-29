import os
import logging
import redis.asyncio as redis
from typing import Dict, Any
from omegaconf import DictConfig, OmegaConf

# Core Components
from Phoenix_project.context_bus import ContextBus
from Phoenix_project.api.gemini_pool_manager import GeminiPoolManager
from Phoenix_project.ai.prompt_manager import PromptManager
from Phoenix_project.ai.prompt_renderer import PromptRenderer

logger = logging.getLogger(__name__)

class PhoenixFactory:
    """
    [主人喵 Factory] 依赖注入工厂。
    确保 Brain (PhoenixProject) 和 Body (Worker) 共享相同的组件构造逻辑。
    """

    @staticmethod
    def create_redis_client() -> redis.Redis:
        host = os.environ.get("REDIS_HOST", "localhost")
        port = int(os.environ.get("REDIS_PORT", 6379))
        # [Task 1.3] 返回异步 Redis 客户端
        return redis.Redis(host=host, port=port, db=0, decode_responses=False)

    @staticmethod
    def create_context_bus(redis_client: redis.Redis, config: Dict[str, Any]) -> ContextBus:
        return ContextBus(redis_client=redis_client, config=config)

    @staticmethod
    def create_common_services(cfg: DictConfig) -> Dict[str, Any]:
        """
        创建共享的基础设施服务。
        """
        redis_client = PhoenixFactory.create_redis_client()
        
        # 将 DictConfig 转换为原生字典
        context_bus_cfg = OmegaConf.to_container(cfg.context_bus, resolve=True) if 'context_bus' in cfg else {}
        context_bus = PhoenixFactory.create_context_bus(redis_client, context_bus_cfg)

        # 基础管理器
        prompt_manager = PromptManager(prompt_directory="prompts")
        prompt_renderer = PromptRenderer(prompt_manager=prompt_manager)
        
        gemini_manager = GeminiPoolManager(config=cfg.api_gateway) if 'api_gateway' in cfg else None
        
        return {
            "redis_client": redis_client,
            "context_bus": context_bus,
            "prompt_manager": prompt_manager,
            "prompt_renderer": prompt_renderer,
            "gemini_manager": gemini_manager
        }

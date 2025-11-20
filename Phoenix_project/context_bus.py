import json
import redis
from typing import Optional, Dict, Any

# 修复：将相对导入 'from .core.pipeline_state...' 更改为绝对导入
from Phoenix_project.core.pipeline_state import PipelineState
# 修复：将相对导入 'from .monitor.logging...' 更改为绝对导入
from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

class ContextBus:
    """
    [Refactored Phase 2.2]
    状态持久化适配器 (State Persistence Adapter)。
    负责将 PipelineState 序列化并存储到 Redis，以及从 Redis 恢复状态。
    不再维护单例，也不在内存中持有锁（并发由 Orchestrator 控制）。
    """

    def __init__(self, redis_client: redis.Redis, config: Optional[Dict[str, Any]] = None):
        self.redis = redis_client
        self.config = config or {}
        # 默认过期时间: 24小时
        self.ttl = self.config.get("state_ttl_sec", 86400)

    def save_state(self, state: PipelineState) -> bool:
        """
        将当前状态快照持久化到 Redis。
        Key: phoenix:state:{run_id}
        """
        try:
            key = f"phoenix:state:{state.run_id}"
            # Pydantic model_dump_json handles serialization
            json_data = state.model_dump_json()
            
            # 存储状态并设置过期时间
            self.redis.setex(key, self.ttl, json_data)
            
            # 同时更新一个 "latest" 指针，方便调试或恢复最近一次运行
            self.redis.set("phoenix:state:latest_run_id", state.run_id)
            
            logger.debug(f"State saved to Redis. RunID: {state.run_id}, Step: {state.step_index}")
            return True
        except Exception as e:
            logger.error(f"Failed to save state to Redis: {e}", exc_info=True)
            return False

    def load_state(self, run_id: str) -> Optional[PipelineState]:
        """
        从 Redis 加载指定 RunID 的状态。
        """
        try:
            key = f"phoenix:state:{run_id}"
            data = self.redis.get(key)
            
            if not data:
                logger.warning(f"No state found for RunID: {run_id}")
                return None
                
            # Pydantic 反序列化
            state = PipelineState.model_validate_json(data)
            logger.info(f"State loaded for RunID: {run_id}, Step: {state.step_index}")
            return state
        except Exception as e:
            logger.error(f"Failed to load state for RunID {run_id}: {e}", exc_info=True)
            return None

    def load_latest_state(self) -> Optional[PipelineState]:
        """
        尝试恢复最近一次运行的状态。
        """
        try:
            run_id = self.redis.get("phoenix:state:latest_run_id")
            if run_id:
                if isinstance(run_id, bytes):
                    run_id = run_id.decode('utf-8')
                return self.load_state(run_id)
            return None
        except Exception as e:
            logger.error(f"Failed to load latest state: {e}", exc_info=True)
            return None

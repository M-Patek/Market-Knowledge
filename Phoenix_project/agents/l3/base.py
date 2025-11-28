# agents/l3/base.py
from abc import ABC, abstractmethod
import asyncio
import numpy as np
from typing import Optional, Type, List, Any

# [任务 4.1] 导入 RLLib 核心组件
from ray.rllib.algorithms.algorithm import Algorithm
from Phoenix_project.core.schemas.fusion_result import FusionResult
from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

class BaseDRLAgent(ABC):
    """
    L3 DRL 智能体的抽象基类。
    
    [任务 4.1] 更新:
    - __init__ 现在存储加载的 RLLib Algorithm。
    - 实现了 compute_action 方法 (Task 0.2 Async)。
    """
    
    def __init__(self, algorithm: Algorithm):
        """
        [任务 4.1] 构造函数现在接收一个已加载的 RLLib 算法实例。
        """
        if not isinstance(algorithm, Algorithm):
            raise ValueError(f"Expected ray.rllib.algorithms.algorithm.Algorithm, got {type(algorithm)}")
        self.algorithm = algorithm
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info(f"DRL Agent ({self.__class__.__name__}) initialized with algorithm.")
        
        # [Task 3.1] Initialize Internal Memory (Hidden States)
        # Required for LSTM/GRU to maintain context across time steps.
        try:
            self.internal_state = self.algorithm.get_policy().get_initial_state()
        except Exception as e:
            self.logger.warning(f"State initialization failed (defaulting to stateless): {e}")
            self.internal_state = []

    @abstractmethod
    def _format_obs(self, state_data: dict, fusion_result: Optional[FusionResult]) -> np.ndarray:
        """
        (抽象方法)
        子类 (Alpha, Risk, Exec) 必须实现此方法。
        它将 Phoenix 状态 转换为 匹配 TradingEnv 的 np.ndarray。
        """
        pass
    
    @abstractmethod
    def get_safe_action(self) -> np.ndarray:
        """
        [Safety Phase II] Returns a safe fallback action in case of inference failure.
        Must be implemented by subclasses.
        """
        pass

    def format_observation(self, state_data: dict, fusion_result: Optional[FusionResult]) -> np.ndarray:
        """
        (公共包装器)
        调用子类的 _format_obs 方法。
        """
        return self._format_obs(state_data, fusion_result)

    async def compute_action(self, observation: np.ndarray) -> np.ndarray:
        """
        [任务 4.1 & 0.2] 完成“神经连接”。
        使用加载的 RLLib 算法计算确定性动作。
        
        [Fix 0.2] 使用 asyncio.to_thread 防止阻塞主事件循环。
        (由 Orchestrator 调用)
        """
        try:
            # explore=False 确保我们在生产 (Inference) 模式下
            # 获取确定性动作，而不是随机探索。
            # [Task 3.1] Enable Stateful Inference (Pass & Update Hidden States)
            result = await asyncio.to_thread(
                self.algorithm.compute_single_action,
                observation,
                state=self.internal_state,
                explore=False
            )
            
            # Unpack Result: (action, state_out, extra_info)
            if isinstance(result, tuple) and len(result) >= 3:
                action, state_out, _ = result
                self.internal_state = state_out
                return action
            
            # Fallback: Stateless model returns just action
            return result
        
        except Exception as e:
            self.logger.error(f"Error during compute_single_action: {e}", exc_info=True)
            # [Safety Phase II] Use agent-specific safe fallback
            return self.get_safe_action()


# [任务 4.1] DRLAgentLoader (由 registry.py 使用)
class DRLAgentLoader:
    """
    一个静态类，负责从检查点路径加载 RLLib Algorithm
    并将其封装到我们的 BaseDRLAgent 子类中。
    """
    
    @staticmethod
    def load_agent(
        agent_class: Type[BaseDRLAgent], # (例如 AlphaAgent, RiskAgent)
        checkpoint_path: str
    ) -> Optional[BaseDRLAgent]:
        """
        [任务 4.1] 实现加载器。
        (由 registry.py 在 [任务 1.1] 中调用)
        
        Args:
            agent_class: 要实例化的智能体类 (例如 AlphaAgent)。
            checkpoint_path: 指向 RLLib 检查点目录的路径 (来自 system.yaml)。

        Returns:
            一个已初始化的、准备好用于推理的 BaseDRLAgent 实例。
        """
        if not checkpoint_path:
            logger.error(f"Cannot load {agent_class.__name__}: checkpoint_path is missing or empty.")
            return None
            
        logger.info(f"Loading {agent_class.__name__} from checkpoint: {checkpoint_path}...")
        
        try:
            # 1. (关键) 从 RLLib 检查点恢复算法
            algo = Algorithm.from_checkpoint(checkpoint_path)
            logger.info(f"RLLib Algorithm loaded successfully for {agent_class.__name__}.")
            
            # 2. 将加载的算法注入到我们的智能体包装器中
            agent_instance = agent_class(algorithm=algo)
            
            logger.info(f"{agent_class.__name__} instance created and ready.")
            return agent_instance
            
        except FileNotFoundError:
            logger.error(f"FATAL: Checkpoint directory not found at: {checkpoint_path}")
        except Exception as e:
            logger.error(f"FATAL: Failed to load algorithm from checkpoint {checkpoint_path}. Error: {e}", exc_info=True)
            
        return None

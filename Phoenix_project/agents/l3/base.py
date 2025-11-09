# agents/l3/base.py
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional, Type

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
    - 实现了 compute_action 方法。
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

    @abstractmethod
    def _format_obs(self, state_data: dict, fusion_result: Optional[FusionResult]) -> np.ndarray:
        """
        (抽象方法)
        子类 (Alpha, Risk, Exec) 必须实现此方法。
        它将 Phoenix 状态 转换为 匹配 TradingEnv 的 np.ndarray。
        """
        pass

    def format_observation(self, state_data: dict, fusion_result: Optional[FusionResult]) -> np.ndarray:
        """
        (公共包装器)
        调用子类的 _format_obs 方法。
        """
        return self._format_obs(state_data, fusion_result)

    def compute_action(self, observation: np.ndarray) -> np.ndarray:
        """
        [任务 4.1] 完成“神经连接”。
        使用加载的 RLLib 算法计算确定性动作。
        
        (由 Orchestrator 在 [任务 2.3] 中调用)
        """
        try:
            # explore=False 确保我们在生产 (Inference) 模式下
            # 获取确定性动作，而不是随机探索。
            action = self.algorithm.compute_single_action(
                observation,
                explore=False
            )
            return action
        
        except Exception as e:
            self.logger.error(f"Error during compute_single_action: {e}", exc_info=True)
            # (根据 agent 类型返回一个安全的默认动作)
            return np.array([0.0]) # (例如 0.0 权重 / 0.0 风险 / 'Hold' 动作)


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
        (由 registry.py 在 [任务 2.2] 中调用)
        
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

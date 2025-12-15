# agents/l3/base.py
from abc import ABC, abstractmethod
import asyncio
import numpy as np
from typing import Optional, Type, List, Any, Dict

# [任务 4.1] 导入 RLLib 核心组件
from ray.rllib.algorithms.algorithm import Algorithm
from Phoenix_project.core.schemas.fusion_result import FusionResult
from Phoenix_project.monitor.logging import get_logger
from Phoenix_project.core.exceptions import CriticalConfigurationError # [TASK-P2-003 Fix] Import new exception

logger = get_logger(__name__)

class BaseDRLAgent(ABC):
    """
    L3 DRL 智能体的抽象基类。
    
    [任务 4.1] 更新:
    - __init__ 现在存储加载的 RLLib Algorithm。
    - 实现了 compute_action 方法 (Task 0.2 Async)。
    
    [Beta FIX] Removed Procrustean Bed (Fixed Dimension).
    Now dynamically adapts to the model's observation space.
    """
    
    def __init__(self, algorithm: Algorithm, policy_id: str = "default_policy", context_bus: Any = None, audit_manager: Any = None):
        """
        [任务 4.1] 构造函数现在接收一个已加载的 RLLib 算法实例。
        [Task 3.3] 增加 context_bus 和 audit_manager 参数支持，以便子类记录决策日志。
        """
        if not isinstance(algorithm, Algorithm):
            raise ValueError(f"Expected ray.rllib.algorithms.algorithm.Algorithm, got {type(algorithm)}")
        self.algorithm = algorithm
        self.policy_id = policy_id # [Beta FIX] Bind to specific policy
        self.logger = get_logger(self.__class__.__name__)
        
        # [Task 3.3] Initialize dependencies
        self.context_bus = context_bus
        self.audit_manager = audit_manager
        
        # [Beta FIX] Dynamic Dimension Detection
        try:
            policy = self.algorithm.get_policy(self.policy_id)
            if policy:
                self.expected_dim = policy.observation_space.shape[0]
                self.logger.info(f"DRL Agent initialized. Policy: {self.policy_id}, Expected Obs Dim: {self.expected_dim}")
                self.internal_state = policy.get_initial_state()
            else:
                # [TASK-P2-003 Fix 2] Critical failure if policy is missing
                error_msg = f"CRITICAL: Policy '{self.policy_id}' not found in algorithm. Cannot initialize DRL Agent."
                self.logger.critical(error_msg)
                raise CriticalConfigurationError(error_msg)
        except CriticalConfigurationError:
             raise # Re-raise if policy not found
        except Exception as e:
            # [TASK-P2-003 Fix 2] Critical failure on dimension/policy inspection
            error_msg = f"CRITICAL: Failed to inspect policy or initialize state for DRL Agent. Error: {e}"
            self.logger.critical(error_msg)
            raise CriticalConfigurationError(error_msg)

    def set_dependencies(self, context_bus: Any, audit_manager: Any):
        """
        [Task 2.3] 允许手动注入核心依赖 (ContextBus, AuditManager)。
        打破 L3 通信孤岛。
        """
        self.context_bus = context_bus
        self.audit_manager = audit_manager
        self.logger.info(f"Dependencies injected into {self.__class__.__name__} (Bus: {bool(context_bus)}, Audit: {bool(audit_manager)})")

    @abstractmethod
    def _format_obs(self, state_data: dict, fusion_result: Optional[FusionResult], market_state: Optional[Dict[str, Any]] = None) -> np.ndarray:
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

    def format_observation(self, state_data: dict, fusion_result: Optional[FusionResult], market_state: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        (公共包装器)
        调用子类的 _format_obs 方法。
        """
        return self._format_obs(state_data, fusion_result, market_state)

    async def compute_action(self, observation: np.ndarray) -> np.ndarray:
        """
        [任务 4.1 & 0.2] 完成“神经连接”。
        使用加载的 RLLib 算法计算确定性动作。
        """
        # [TASK-P2-003 Fix 1] Removed the soft/adaptive observation shaping logic.
        # Enforce dimension check here, relying on RLLib to throw an error if it fails,
        # or throwing a critical error if the expected dimension exists and the input doesn't match.
        if hasattr(self, 'expected_dim') and self.expected_dim and observation.shape[0] != self.expected_dim:
             error_msg = f"CRITICAL: Observation dimension mismatch! Expected {self.expected_dim}, got {observation.shape[0]}. Aborting."
             self.logger.critical(error_msg)
             raise CriticalConfigurationError(error_msg)

        try:
            # explore=False 确保我们在生产 (Inference) 模式下
            # 获取确定性动作，而不是随机探索。
            result = await asyncio.to_thread(
                self.algorithm.compute_single_action,
                observation,
                state=self.internal_state,
                policy_id=self.policy_id, # [Beta FIX] Use correct policy
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
            safe_action = self.get_safe_action()
            return safe_action


# [任务 4.1] DRLAgentLoader (由 registry.py 使用)
class DRLAgentLoader:
    """
    一个静态类，负责从检查点路径加载 RLLib Algorithm
    并将其封装到我们的 BaseDRLAgent 子类中。
    """
    
    @staticmethod
    def load_agent(
        agent_class: Type[BaseDRLAgent], 
        checkpoint_path: str,
        policy_id: str = "default_policy" # [Beta FIX] Support Policy ID
    ) -> Optional[BaseDRLAgent]:
        """
        [任务 4.1] 实现加载器。
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
            agent_instance = agent_class(algorithm=algo, policy_id=policy_id)
            
            logger.info(f"{agent_class.__name__} instance created and ready.")
            return agent_instance
            
        except FileNotFoundError:
            logger.error(f"FATAL: Checkpoint directory not found at: {checkpoint_path}")
        except CriticalConfigurationError as e:
             # Propagate critical error raised during DRLAgent __init__
             logger.error(f"FATAL: Critical configuration error during agent initialization: {e}")
             return None
        except Exception as e:
            logger.error(f"FATAL: Failed to load algorithm from checkpoint {checkpoint_path}. Error: {e}", exc_info=True)
            
        return None

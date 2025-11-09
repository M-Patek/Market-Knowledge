import numpy as np
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from typing import Optional, Dict, Any, Type

# 移除旧的 SB3 导入
# from stable_baselines3 import PPO

from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

class BaseL3Agent:
    """
    [MARL 重构]
    L3 DRL 智能体的基类，使用 Ray RLLib 进行模型加载和推理。

    它使用一个类变量 `_algorithm` 来确保 RLLib 检查点只被加载一次，
    并由所有 L3 智能体实例（Alpha, Risk, Execution）共享。
    """

    # [新增] 使用类变量来共享 RLLib 算法实例
    _algorithm: Optional[Algorithm] = None
    _model_path: Optional[str] = None
    _algorithm_class: Type = Algorithm # 默认为 Algorithm，可以被覆盖

    def __init__(self, config: Dict[str, Any], policy_id: str):
        """
        [重构] __init__

        Args:
            config (Dict[str, Any]): 包含 'model_path' (RLLib 检查点目录) 的配置
            policy_id (str): 此智能体应使用的策略 ID (例如 "alpha_policy")
        """
        self.config = config
        self.policy_id = policy_id

        # [重构] 确保模型已加载
        # 'model_path' 现在应该是检查点目录
        model_path = self.config.get('model_path')
        if not model_path:
            logger.error(f"Agent {self.policy_id} 未配置 'model_path'")
            raise ValueError(f"Agent {self.policy_id} 未配置 'model_path'")

        # 使用 self.__class__ 来调用类方法
        self.__class__._load_rllib_model(model_path)

        # 移除旧的 SB3 模型加载
        # self.model = self.load_model(model_path)

    @classmethod
    def _load_rllib_model(cls, model_path: str):
        """
        [新增] 
        一个类方法，用于初始化 Ray 并从检查点加载 RLLib 算法。
        这确保了它在所有 L3 智能体中只执行一次。
        """
        # 如果模型已加载且路径相同，则跳过
        if cls._algorithm and cls._model_path == model_path:
            return

        logger.info(f"首次加载 RLLib 模型，路径: {model_path}...")

        # 1. 初始化 Ray (如果尚未初始化)
        if not ray.is_initialized():
            try:
                ray.init(
                    logging_level="ERROR", 
                    ignore_reinit_error=True,
                    # (可选) 确保为推理分配足够资源
                    num_cpus=1 
                )
                logger.debug("Ray (for inference) initialized.")
            except Exception as e:
                logger.warning(f"Ray init failed (maybe already init?): {e}")


        # 2. 从检查点恢复算法
        try:
            # [关键] 我们使用 Algorithm.from_checkpoint
            # 这将恢复完整的训练器状态（包括策略）
            cls._algorithm = cls._algorithm_class.from_checkpoint(model_path)
            cls._model_path = model_path
            logger.info(f"RLLib 算法已从 {model_path} 成功恢复。")
        except Exception as e:
            logger.error(f"加载 RLLib 检查点失败: {e}", exc_info=True)
            raise

    def predict(self, observation: np.ndarray) -> int:
        """
        [重构] 
        使用加载的 RLLib 算法执行推理。

        Args:
            observation (np.ndarray): 格式必须与 TradingEnv 中定义的一致
                                      (例如 [balance, shares_held, current_price])

        Returns:
            int: 动作 (例如 0, 1, 2)
        """
        if not self._algorithm:
            logger.error(f"RLLib 算法未加载 (Policy: {self.policy_id})。")
            raise RuntimeError("RLLib 算法未加载。请先调用 load_model。")

        try:
            # [关键] 使用 compute_single_action
            action = self._algorithm.compute_single_action(
                observation=observation,
                policy_id=self.policy_id,
                # 禁用探索，确保确定性
                explore=False 
            )

            # (RLLib 通常返回 numpy 类型，确保是 int)
            return int(action)

        except Exception as e:
            logger.error(f"RLLib 推理失败 (Policy: {self.policy_id}): {e}", exc_info=True)
            # 失败时返回安全动作（例如 "持有"）
            return 1 # 假设 1 是 "持有"

    # [移除] 旧的 SB3 load_model
    # def load_model(self, model_path: str):
    #     if not model_path:
    #         logger.warning(f"No model path provided for {self.__class__.__name__}.")
    #         return None
    #     try:
    #         return PPO.load(model_path)
    #     except Exception as e:
    #         logger.error(f"Failed to load DRL model from {model_path}: {e}", exc_info=True)
    #         return None

    def execute(self, state_data: Dict[str, Any]) -> int:
        """
        [重构]
        子类必须实现此方法以格式化 obs 并调用 self.predict()。
        """
        raise NotImplementedError("L3 智能体必须实现 execute 方法")

    def _format_obs(self, state_data: Dict[str, Any]) -> np.ndarray:
        """
        [新增]
        子类必须实现此方法以匹配 TradingEnv 的状态。
        """
        raise NotImplementedError("L3 智能体必须实现 _format_obs 方法")

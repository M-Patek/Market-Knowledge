from typing import Dict, Any
import numpy as np
from .base import BaseL3Agent
from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

class RiskAgent(BaseL3Agent):
    """
    [MARL 重构]
    Risk 智能体，使用 RLLib 基类进行推理。
    负责决定 (批准/否决)。
    """
    def __init__(self, config: Dict[str, Any]):
        """
        [重构]
        初始化 RiskAgent。
        """
        # [关键] 传递 "risk_policy"
        super().__init__(config, policy_id="risk_policy")

    def execute(self, state_data: Dict[str, Any]) -> int:
        """
        [重构]
        从 PipelineState 获取数据，格式化为观测值，并使用 RLLib predict。
        """
        # 1. 格式化观测数据
        try:
            obs = self._format_obs(state_data)
        except Exception as e:
            logger.error(f"RiskAgent: 状态格式化失败: {e}", exc_info=True)
            return 0 # 返回安全动作 (批准)

        # 2. 调用基类的 RLLib predict
        action = self.predict(obs) 
        
        # 0=批准, 1=否决
        return action

    def _format_obs(self, state_data: Dict[str, Any]) -> np.ndarray:
        """
        [新增]
        将 PipelineState 字典转换为 TradingEnv 所需的 np.array 状态。
        
        [重要] 这必须与 PettingZoo TradingEnv 中的 _get_state() 方法完全一致。
        (状态: [balance, shares_held, current_price])
        
        注意: 即使 RiskAgent 只关心风险，它也被训练为
        查看与 Alpha 相同的状态。
        """
        try:
            balance = float(state_data.get('current_balance', 10000.0))
            shares_held = int(state_data.get('shares_held', 0))
            current_price = float(state_data.get('current_price', 0.0))

            if current_price == 0.0:
                logger.warning("RiskAgent: 观测数据中当前价格为 0")

            # 必须匹配 TradingEnv 的 DType (np.float32)
            return np.array([
                balance,
                shares_held,
                current_price
            ], dtype=np.float32)
        
        except Exception as e:
            logger.error(f"RiskAgent: _format_obs 失败: {e}", exc_info=True)
            # 返回一个安全的默认状态
            return np.array([10000.0, 0, 0.0], dtype=np.float32)

from typing import Dict, Any, Optional
import numpy as np
from .base import BaseDRLAgent
from Phoenix_project.monitor.logging import get_logger
from Phoenix_project.core.schemas.fusion_result import FusionResult

logger = get_logger(__name__)

class RiskAgent(BaseDRLAgent):
    """
    [MARL 重构]
    Risk 智能体，使用 RLLib 基类进行推理。
    负责决定 (批准/否决)。
    """
    
    def get_safe_action(self) -> np.ndarray:
        """
        [Safety] Default to HALT (1.0) on failure.
        Action Space: Discrete(2) or Continuous(1) -> >0.5 means HALT.
        """
        return np.array([1.0])
    
    def _format_obs(self, state_data: dict, fusion_result: Optional[FusionResult]) -> np.ndarray:
        """
        [任务 2.1] 格式化观察值以匹配 TradingEnv 的新 (5-d) 状态空间。
        
        Args:
            state_data (dict): 包含 {'balance', 'holdings', 'price'} 的实时数据。
            fusion_result (FusionResult): 来自 L2 认知引擎的分析结果。

        Returns:
            np.ndarray: 匹配 TradingEnv.observation_space 的 5-d 状态向量。
        """
        # 1. 从 state_data 中提取市场状态
        balance = state_data.get('balance', 0.0)
        holdings = state_data.get('holdings', 0.0)
        price = state_data.get('price', 0.0)

        # 2. (关键) 从 L2 FusionResult 中提取 L2 特征
        if fusion_result:
            # [Fix] Map L2 decision string to numeric sentiment for RL observation
            decision_str = str(getattr(fusion_result, "decision", "HOLD")).upper()
            score_map = {"STRONG_BUY": 1.0, "BUY": 0.5, "SELL": -0.5, "STRONG_SELL": -1.0}
            sentiment = score_map.get(decision_str, 0.0)
            confidence = float(getattr(fusion_result, "confidence", 0.5))
        else:
            # 如果没有 L2 结果 (例如周期开始时)，提供默认值
            sentiment = 0.0  # 中性情感
            confidence = 0.5 # 中性信心

        # 3. 构建与 TradingEnv._get_state() 完全匹配的状态向量
        # 状态 (5-d): [balance, holdings, price, l2_sentiment, l2_confidence]
        obs = np.array([
            balance,
            holdings,
            price,
            sentiment,
            confidence
        ], dtype=np.float32)
        
        return obs

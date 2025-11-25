# agents/l3/alpha_agent.py
import numpy as np
from typing import Optional

from .base import BaseDRLAgent
from Phoenix_project.core.schemas.fusion_result import FusionResult

class AlphaAgent(BaseDRLAgent):
    """
    L3 Alpha 智能体。
    负责根据 L2 分析和市场状态，决定理想的 *目标仓位* (例如 目标权重)。
    """

    def get_safe_action(self) -> np.ndarray:
        """
        [Safety] Default to NEUTRAL (0.0) allocation on failure.
        """
        return np.array([0.0])

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
        sentiment = 0.0
        confidence = 0.5
        
        if fusion_result:
            # 映射字符串决策到数值情感
            decision_map = {
                "STRONG_BUY": 1.0, 
                "BUY": 0.5, 
                "HOLD": 0.0, "NEUTRAL": 0.0,
                "SELL": -0.5, 
                "STRONG_SELL": -1.0
            }
            # 获取 decision 字段，默认 HOLD
            decision_str = getattr(fusion_result, 'decision', 'HOLD')
            sentiment = decision_map.get(str(decision_str).upper(), 0.0)
            
            # 获取 confidence 字段
            confidence = getattr(fusion_result, 'confidence', 0.5)

        # [Task 2.1] Normalize Inputs to prevent gradient vanishing
        norm_balance = np.log(balance + 1.0)
        norm_price = np.log(price + 1e-9) if price > 0 else 0.0

        # 3. 构建与 TradingEnv._get_state() 完全匹配的状态向量
        # 状态 (5-d): [norm_balance, holdings, norm_price, sentiment, confidence]
        obs = np.array([
            norm_balance,
            holdings,
            norm_price,
            sentiment,
            confidence
        ], dtype=np.float32)
        
        return obs

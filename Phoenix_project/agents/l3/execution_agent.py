# agents/l3/execution_agent.py
import numpy as np
from typing import Optional

from .base import BaseDRLAgent
from Phoenix_project.core.schemas.fusion_result import FusionResult

class ExecutionAgent(BaseDRLAgent):
    """
    L3 Execution 智能体。
    负责根据 L2 分析和市场状态，决定执行策略（如 TWAP, VWAP 等，或简单的执行强度）。
    """
    
    def get_safe_action(self) -> np.ndarray:
        """
        [Safety] Return None to signal Orchestrator to HALT/HOLD.
        """
        return None

    def _format_obs(self, state_data: dict, fusion_result: Optional[FusionResult]) -> np.ndarray:
        """
        [Task 3.3] Format observation to match TradingEnv (6-d).
        Vector: [NormBalance, PositionWeight, LogReturn, LogVolume, Sentiment, Confidence]
        
        Args:
            state_data (dict): {'balance', 'initial_balance', 'position_weight', 'price', 'prev_price', 'volume'}
            fusion_result (FusionResult): 来自 L2 认知引擎的分析结果。

        Returns:
            np.ndarray: (6,) float32 vector.
        """
        # 1. 从 state_data 中提取市场状态
        balance = state_data.get('balance', 0.0)
        initial_balance = state_data.get('initial_balance', balance)
        position_weight = state_data.get('position_weight', 0.0)
        price = state_data.get('price', 0.0)
        prev_price = state_data.get('prev_price', price)
        volume = state_data.get('volume', 0.0)

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

        # [Task 3.3] Feature Engineering
        norm_balance = (balance / initial_balance) if initial_balance > 0 else 1.0
        
        log_return = 0.0
        if price > 0 and prev_price > 0:
            log_return = np.log(price / prev_price)
            
        log_volume = np.log(volume + 1.0)

        # 3. Construct 6-d State Vector
        obs = np.array([
            norm_balance,
            position_weight,
            log_return,
            log_volume,
            sentiment,
            confidence
        ], dtype=np.float32)
        
        return obs

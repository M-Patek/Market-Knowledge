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
        [Safety] Returns neutral zero vector.
        Fixed: Must return np.ndarray, not None.
        """
        return np.array([0.0], dtype=np.float32)
    
    def _format_obs(self, state_data: dict, fusion_result: Optional[FusionResult], market_state: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        [Task 6.1] Format observation to match TradingEnv (7-d).
        Vector: [NormBalance, PositionWeight, LogReturn, LogVolume, Sentiment, Confidence, MarketRegime]
        
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
            # [Fix] Map L2 decision string to numeric sentiment for RL observation
            decision_str = str(getattr(fusion_result, "decision", "HOLD")).upper()
            score_map = {"STRONG_BUY": 1.0, "BUY": 0.5, "SELL": -0.5, "STRONG_SELL": -1.0}
            sentiment = score_map.get(decision_str, 0.0)
            confidence = float(getattr(fusion_result, "confidence", 0.5))

        # [Task 3.3] Feature Engineering
        norm_balance = (balance / initial_balance) if initial_balance > 0 else 1.0
        
        log_return = 0.0
        if price > 0 and prev_price > 0:
            log_return = np.log(price / prev_price)
            
        log_volume = np.log(volume + 1.0)

        # [Task 6.1] Macro Feature
        regime_val = 0.0
        if market_state:
            regime = str(market_state.get('regime', 'NEUTRAL')).upper()
            if 'BULL' in regime: regime_val = 1.0
            elif 'BEAR' in regime: regime_val = -1.0

        # 3. Construct 7-d State Vector
        obs = np.array([
            norm_balance,
            position_weight,
            log_return,
            log_volume,
            sentiment,
            confidence,
            regime_val
        ], dtype=np.float32)
        
        return obs

# agents/l3/execution_agent.py
import numpy as np
from typing import Optional, Dict, Any

from .base import BaseDRLAgent
from Phoenix_project.core.schemas.fusion_result import FusionResult

class ExecutionAgent(BaseDRLAgent):
    """
    L3 Execution 智能体。
    负责根据 L2 分析和市场状态，决定执行策略。
    [Beta FIX] Aligned with 9-dim observation space.

    Action Space Semantics [Task P2-RISK-04]:
    Continuous scalar [-1.0, 1.0]:
    - Positive (0.0 to 1.0): Buy / Long Exposure Intensity.
    - Negative (-1.0 to 0.0): Sell / Short Exposure Intensity.
    - 0.0: Hold / Neutral.
    Values are interpreted as target weight adjustment or execution urgency depending on the TradingEnv configuration.
    """
    
    def get_safe_action(self) -> np.ndarray:
        """
        [Safety] Returns neutral zero vector (Hold).
        """
        return np.array([0.0], dtype=np.float32)

    async def compute_action(self, observation: np.ndarray) -> np.ndarray:
        """
        [Task 4.1] Standard inference logic with [Task P2-RISK-04] Semantic Validation.
        """
        raw_action = await super().compute_action(observation)
        
        # [Task P2-RISK-04] Sanity Check & Clamping
        # Prevent extreme outliers from DRL model (e.g., > 1.0 leverage requests).
        # We enforce strict bounds [-1.0, 1.0].
        clipped_action = np.clip(raw_action, -1.0, 1.0)
        
        if not np.allclose(raw_action, clipped_action, atol=1e-5):
             # Use self.logger inherited from BaseDRLAgent
             self.logger.warning(
                f"L3 Execution Action outlier detected: {raw_action}. "
                f"Clamped to {clipped_action} to enforce safety limits."
            )
             
        return clipped_action

    def _format_obs(self, state_data: dict, fusion_result: Optional[FusionResult], market_state: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        [Beta FIX] Confirmed 9-Dimensions.
        Vector: [NormBalance, PositionWeight, LogReturn, LogVolume, Sentiment, Confidence, Regime, Spread, Imbalance]
        """
        # 1. State Data
        balance = state_data.get('balance', 0.0)
        initial_balance = state_data.get('initial_balance', balance)
        position_weight = state_data.get('position_weight', 0.0)
        price = state_data.get('price', 0.0)
        prev_price = state_data.get('prev_price', price)
        volume = state_data.get('volume', 0.0)
        
        # Microstructure
        spread = state_data.get('spread', 0.0)
        depth_imbalance = state_data.get('depth_imbalance', 0.0)

        # 2. L2 Features
        sentiment = 0.0
        confidence = 0.5
        
        if fusion_result:
            decision_map = {
                "STRONG_BUY": 1.0, 
                "BUY": 0.5, 
                "HOLD": 0.0, "NEUTRAL": 0.0,
                "SELL": -0.5, 
                "STRONG_SELL": -1.0
            }
            decision_str = getattr(fusion_result, 'decision', 'HOLD')
            sentiment = decision_map.get(str(decision_str).upper(), 0.0)
            confidence = getattr(fusion_result, 'confidence', 0.5)

        # 3. Calculations
        norm_balance = (balance / initial_balance) if initial_balance > 0 else 1.0
        
        log_return = 0.0
        if price > 0 and prev_price > 0:
            log_return = np.log(price / prev_price)
            
        log_volume = np.log(volume + 1.0)

        regime_val = 0.0
        if market_state:
            regime = str(market_state.get('regime', 'NEUTRAL')).upper()
            if 'BULL' in regime: regime_val = 1.0
            elif 'BEAR' in regime: regime_val = -1.0

        # 4. Construct 9-d State Vector
        obs = np.array([
            norm_balance,
            position_weight,
            log_return,
            log_volume,
            sentiment,
            confidence,
            regime_val,
            spread,
            depth_imbalance
        ], dtype=np.float32)
        
        return obs

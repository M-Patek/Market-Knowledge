# agents/l3/alpha_agent.py
import numpy as np
from typing import Optional, Dict, Any

from .base import BaseDRLAgent
from Phoenix_project.core.schemas.fusion_result import FusionResult

class AlphaAgent(BaseDRLAgent):
    """
    L3 Alpha 智能体。
    负责根据 L2 分析和市场状态，决定理想的 *目标仓位* (例如 目标权重)。
    [Beta FIX] Unified 9-dim observation space.
    """

    def get_safe_action(self) -> np.ndarray:
        """
        [Safety] Returns neutral zero vector.
        [Beta FIX] Shape-aware fallback (Dynamically based on action space if possible, else default to 1D).
        Note: Ideally we inspect self.algorithm.action_space, but for now we return a broadcastable zero.
        """
        # Return scalar 0.0 which can often broadcast, or handled by downstream validation.
        # Ideally, we should know the asset universe size.
        return np.array([0.0], dtype=np.float32)

    async def compute_action(self, observation: np.ndarray) -> np.ndarray:
        """
        [Task 3.3] Override to apply safety normalization.
        """
        try:
            raw_action = await super().compute_action(observation)
            
            if raw_action is None:
                return self.get_safe_action()
                
            # Normalize
            if raw_action.size == 1:
                return np.clip(raw_action, -1.0, 1.0)
                
            # [Beta Note] Improved normalization could go here (Softmax etc.)
            # For now keeping L1 but ensuring it doesn't crash on zeros
            total_exposure = np.sum(np.abs(raw_action))
            if total_exposure > 1.0:
                self.logger.warning(f"Clipping high leverage action (Exposure: {total_exposure:.2f}x).")
                return raw_action / total_exposure
                
            return raw_action
            
        except Exception as e:
            self.logger.error(f"Error in AlphaAgent compute_action: {e}")
            return self.get_safe_action()

    def _format_obs(self, state_data: dict, fusion_result: Optional[FusionResult], market_state: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        [Beta FIX] Unified World View (9-Dimensions).
        Matches ExecutionAgent and TrainingEnv V7.
        Vector: [NormBalance, PositionWeight, LogReturn, LogVolume, Sentiment, Confidence, Regime, Spread, Imbalance]
        """
        # 1. State Data
        balance = state_data.get('balance', 0.0)
        initial_balance = state_data.get('initial_balance', balance)
        position_weight = state_data.get('position_weight', 0.0)
        price = state_data.get('price', 0.0)
        prev_price = state_data.get('prev_price', price)
        volume = state_data.get('volume', 0.0)
        
        # [Beta FIX] Added Microstructure features (previously missing in Alpha)
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
        if price > 1e-8 and prev_price > 1e-8:
            log_return = np.log(price / prev_price)
            
        log_volume = np.log(volume + 1.0) if volume >= 0 else 0.0

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
            spread,           # [New]
            depth_imbalance   # [New]
        ], dtype=np.float32)
        
        return obs

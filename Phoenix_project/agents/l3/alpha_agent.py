# agents/l3/alpha_agent.py
import numpy as np
from typing import Optional, Dict, Any

from .base import BaseDRLAgent
from Phoenix_project.core.schemas.fusion_result import FusionResult

class AlphaAgent(BaseDRLAgent):
    """
    L3 Alpha 智能体。
    负责根据 L2 分析和市场状态，决定理想的 *目标仓位* (例如 目标权重)。
    """

    def get_safe_action(self) -> np.ndarray:
        """
        [Safety] Returns neutral zero vector to signal Orchestrator to HOLD.
        Fixed: Must return np.ndarray, not None.
        """
        # Return 1D array with single element 0.0
        return np.array([0.0], dtype=np.float32)

    async def compute_action(self, observation: np.ndarray) -> np.ndarray:
        """
        [Task 3.3] Override to apply safety normalization (L1 Norm).
        Ensures leverage never exceeds 1.0.
        """
        try:
            raw_action = await super().compute_action(observation)
            
            # Safety Check
            if raw_action is None:
                return self.get_safe_action()
                
            # Normalize
            # If scalar (single asset): Clip to [-1, 1]
            if raw_action.size == 1:
                return np.clip(raw_action, -1.0, 1.0)
                
            # If vector (multi asset): L1 Normalize if sum(abs) > 1.0
            total_exposure = np.sum(np.abs(raw_action))
            if total_exposure > 1.0:
                # self.logger is available via BaseDRLAgent
                self.logger.warning(f"Clipping high leverage action (Exposure: {total_exposure:.2f}x).")
                return raw_action / total_exposure
                
            return raw_action
            
        except Exception as e:
            self.logger.error(f"Error in AlphaAgent compute_action: {e}")
            return self.get_safe_action()

    def _format_obs(self, state_data: dict, fusion_result: Optional[FusionResult], market_state: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        [Task 6.1] Format observation to match TradingEnv (7-d).
        Vector: [NormBalance, PositionWeight, LogReturn, LogVolume, Sentiment, Confidence, MarketRegime]
        
        Args:
            state_data (dict): {'balance', 'initial_balance', 'holdings', 'price', 'prev_price', 'volume'}
            fusion_result (FusionResult): 来自 L2 认知引擎的分析结果。
            market_state (dict): Macro regime info.

        Returns:
            np.ndarray: (7,) float32 vector.
        """
        # 1. 从 state_data 中提取市场状态
        balance = state_data.get('balance', 0.0)
        initial_balance = state_data.get('initial_balance', balance) # Avoid div/0 if missing
        position_weight = state_data.get('position_weight', 0.0) # Replaces raw holdings
        price = state_data.get('price', 0.0)
        prev_price = state_data.get('prev_price', price) # Default to price -> 0 return
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

        # [Task 3.3] Feature Engineering (Match TradingEnv)
        norm_balance = (balance / initial_balance) if initial_balance > 0 else 1.0
        
        log_return = 0.0
        if price > 0 and prev_price > 0:
            log_return = np.log(price / prev_price)
            
        log_volume = np.log(volume + 1.0)

        # [Task 6.1] Macro Feature: Market Regime
        regime_val = 0.0
        if market_state:
            regime = str(market_state.get('regime', 'NEUTRAL')).upper()
            if 'BULL' in regime: 
                regime_val = 1.0
            elif 'BEAR' in regime: 
                regime_val = -1.0
        
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

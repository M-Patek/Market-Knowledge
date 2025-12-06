from typing import Dict, Any, Optional
import numpy as np
from .base import BaseDRLAgent
from Phoenix_project.monitor.logging import get_logger
from Phoenix_project.core.schemas.fusion_result import FusionResult, SystemStatus

logger = get_logger(__name__)

class RiskAgent(BaseDRLAgent):
    """
    [MARL 重构]
    Risk 智能体，使用 RLLib 基类进行推理。
    负责决定 (批准/否决/紧急停止)。
    """
    
    def get_safe_action(self) -> np.ndarray:
        """
        [Safety Phase II] Returns HALT signal (1.0).
        Fixed: Must return np.ndarray, not None.
        If the agent fails/crashes, we default to HALT (Fail-Closed).
        """
        return np.array([1.0], dtype=np.float32)
    
    async def compute_action(self, observation: np.ndarray, fusion_result: Optional[FusionResult] = None) -> np.ndarray:
        """[Task 4.1] Hard Override: Bypass NN if L2 signals HALT."""
        
        # [Safety Phase III] Check System Health First
        # If the fusion result indicates the system is not OK (e.g. HALT or DEGRADED),
        # we immediately engage the circuit breaker.
        if fusion_result and fusion_result.system_status != SystemStatus.OK:
            logger.warning(f"RiskAgent: Circuit Breaker ENGAGED. System Status: {fusion_result.system_status}")
            return self.get_safe_action()

        # Secondary check for explicit HALT decision string
        if fusion_result and str(getattr(fusion_result, "decision", "")).upper() in ["HALT", "HALT_TRADING"]:
            logger.warning("RiskAgent: Hard stop triggered by L2 signal string.")
            return self.get_safe_action()
            
        return await super().compute_action(observation)
    
    def _format_obs(self, state_data: dict, fusion_result: Optional[FusionResult], market_state: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        [Task 2.1 Fix] Format observation to match TradingEnv (9-d).
        Vector: [NormBalance, PositionWeight, LogReturn, LogVolume, Sentiment, Confidence, MarketRegime, Spread, Imbalance]
        
        Args:
            state_data (dict): {'balance', 'initial_balance', 'position_weight', 'price', 'prev_price', 'volume', 'spread', 'depth_imbalance'}
            fusion_result (FusionResult): 来自 L2 认知引擎的分析结果。
            market_state (dict): Macro regime info.

        Returns:
            np.ndarray: (9,) float32 vector.
        """
        # 1. 从 state_data 中提取市场状态
        balance = state_data.get('balance', 0.0)
        initial_balance = state_data.get('initial_balance', balance)
        position_weight = state_data.get('position_weight', 0.0)
        price = state_data.get('price', 0.0)
        prev_price = state_data.get('prev_price', price)
        volume = state_data.get('volume', 0.0)
        
        # [Task 2.1 Fix] Microstructure Features
        spread = state_data.get('spread', 0.0)
        depth_imbalance = state_data.get('depth_imbalance', 0.0)

        # 2. (关键) 从 L2 FusionResult 中提取 L2 特征
        sentiment = 0.0
        # [Task 2.1 Fix] Neutral Default
        # Default confidence to 0.5 (Neutral) to prevent Cold Start Panic (0.0 often interpreted as high risk)
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

        # [Task 6.1] Macro Feature: Market Regime
        regime_val = 0.0
        if market_state:
            regime = str(market_state.get('regime', 'NEUTRAL')).upper()
            if 'BULL' in regime: regime_val = 1.0
            elif 'BEAR' in regime: regime_val = -1.0

        # 3. Construct 9-d State Vector
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

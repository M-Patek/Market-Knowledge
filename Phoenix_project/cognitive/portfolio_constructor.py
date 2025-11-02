import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

from ..core.pipeline_state import PipelineState
from ..core.schemas.fusion_result import FusionResult, AgentDecision
from ..data_manager import DataManager
# 修复：[FIX-2] 'BaseSizer' 不存在于 'sizing/base.py' 中, 
# 接口被定义为 'IPositionSizer'
from ..sizing.base import IPositionSizer
from .risk_manager import RiskManager

class Portfolio:
    """
    Represents the target state of the portfolio.
    """
    def __init__(self, weights: Dict[str, float], timestamp: pd.Timestamp):
        self.weights = weights # e.g., {"AAPL": 0.5, "MSFT": 0.25, "CASH": 0.25}
        self.timestamp = timestamp
        self.metadata = {}

    def get_target_exposure(self, symbol: str) -> float:
        return self.weights.get(symbol, 0.0)

class PortfolioConstructor:
    """
    Constructs the target portfolio based on the fused AI signal and risk constraints.
    It translates the qualitative AI decision (BUY, SELL, HOLD) into a quantitative
    set of target weights.
    """

    def __init__(self, 
                 config, 
                 data_manager: DataManager, 
                 # 修复：[FIX-2] 更新类型提示
                 sizer: IPositionSizer, 
                 risk_manager: RiskManager):
        """
        Initializes the PortfolioConstructor.

        Args:
            config: The main strategy configuration object.
            data_manager: Client to access historical data for sizing calculations.
            sizer: A Sizing module (e.g., VolatilityParitySizer) to calculate position size.
            risk_manager: The RiskManager to get capital modifiers.
        """
        self.config = config
        self.data_manager = data_manager
        self.sizer = sizer
        self.risk_manager = risk_manager
        
        self.asset_universe = config.get('asset_universe', [])
        self.base_capital = config.get('base_capital', 1000000)
        
    def generate_optimized_portfolio(
        self, 
        state: PipelineState, 
        fusion_result: FusionResult
    ) -> Portfolio:
        """
        Main method to generate the new target portfolio.

        Args:
            state: The current PipelineState (includes current positions).
            fusion_result: The output from the AI cognitive pipeline.

        Returns:
            A Portfolio object with the new target weights.
        """
        
        # 1. Get the capital modifier from the Risk Manager
        # This scales our total market exposure based on AI uncertainty
        capital_modifier = self.risk_manager.calculate_capital_modifier(
            fusion_result.cognitive_uncertainty,
            state
        )
        
        # 2. Determine target symbol(s)
        # For now, we assume the event is for a single symbol
        # A more complex system would handle multi-asset decisions
        target_symbol = self._get_target_symbol(fusion_result, state)
        
        if not target_symbol:
            # If no clear target, hold existing positions or move to cash
            return self._create_hold_portfolio(state)

        # 3. Get the directional signal
        signal_direction = self._get_signal_direction(fusion_result.final_decision)
        
        if signal_direction == 0:
            # AI decision is HOLD
            return self._create_hold_portfolio(state)
            
        # 4. Calculate position size using the Sizer
        # This determines *how much* to buy/sell based on volatility, risk, etc.
        target_size_pct = self.sizer.calculate_target_size(
            symbol=target_symbol,
            state=state,
            signal_strength=fusion_result.final_decision.confidence
        )
        
        # 5. Apply modifiers
        # Target weight = Base Size * Direction * Capital Modifier (Uncertainty)
        target_weight = target_size_pct * signal_direction * capital_modifier
        
        # 6. Construct the final portfolio weights
        # This is a simple single-asset example.
        # A real implementation would optimize across all assets in self.asset_universe
        
        target_weights = {sym: 0.0 for sym in self.asset_universe}
        target_weights[target_symbol] = target_weight
        
        # The remaining weight is allocated to Cash
        cash_weight = 1.0 - abs(target_weight) # Assumes no leverage or shorts > 100%
        target_weights['CASH'] = cash_weight 
        
        portfolio = Portfolio(weights=target_weights, timestamp=state.timestamp)
        portfolio.metadata = {
            'capital_modifier': capital_modifier,
            'signal_direction': signal_direction,
            'base_target_size_pct': target_size_pct,
            'cognitive_uncertainty': fusion_result.cognitive_uncertainty
        }
        
        return portfolio

    def _get_target_symbol(self, fusion_result: FusionResult, state: PipelineState) -> Optional[str]:
        """Utility to determine which asset the AI decision applies to."""
        # Check metadata in the fusion result
        if fusion_result.final_decision.metadata.get('target_symbol'):
            return fusion_result.final_decision.metadata['target_symbol']
        
        # Check event data from pipeline IO
        event = fusion_result.pipeline_io.get('event_data') # Assuming event is stored here
        if event and event.symbols:
            # Just take the first symbol for simplicity
            primary_symbol = event.symbols[0]
            if primary_symbol in self.asset_universe:
                return primary_symbol
                
        return None

    def _get_signal_direction(self, decision: AgentDecision) -> int:
        """Converts AI decision string to a numerical direction."""
        if decision.decision == "BUY":
            return 1
        elif decision.decision == "SELL":
            return -1
        else: # HOLD, ERROR, INVALID_RESPONSE
            return 0

    def _create_hold_portfolio(self, state: PipelineState) -> Portfolio:
        """Creates a portfolio that maintains the current positions."""
        # For simplicity, we'll just create a 100% cash portfolio
        # A real "hold" would maintain existing state.weights
        target_weights = {sym: 0.0 for sym in self.asset_universe}
        target_weights['CASH'] = 1.0
        
        return Portfolio(weights=target_weights, timestamp=state.timestamp)

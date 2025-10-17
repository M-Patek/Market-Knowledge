# cognitive/engine.py
import logging
import pandas as pd
from datetime import date
from typing import List, Dict, Optional, Any

from phoenix_project import StrategyConfig, PositionSizerConfig
from sizing.base import IPositionSizer
from sizing.fixed_fraction import FixedFractionSizer
from sizing.volatility_parity import VolatilityParitySizer
from .risk_manager import RiskManager
from .portfolio_constructor import PortfolioConstructor

class CognitiveEngine:
    def __init__(self, config: StrategyConfig, asset_analysis_data: Optional[Dict[date, Dict]] = None, sentiment_data: Optional[Dict[date, float]] = None, ai_mode: str = "processed"):
        self.config = config
        self.logger = logging.getLogger("PhoenixProject.CognitiveEngine")
        self.risk_manager = RiskManager(config)
        self.portfolio_constructor = PortfolioConstructor(config, asset_analysis_data, mode=config.ai_mode)
        self.position_sizer = self._create_sizer(config.position_sizer)

    def _create_sizer(self, sizer_config: PositionSizerConfig) -> IPositionSizer:
        method = sizer_config.method
        params = sizer_config.parameters
        self.logger.info(f"Initializing position sizer: '{method}' with params: {params}")
        if method == "fixed_fraction":
            return FixedFractionSizer(**params)
        elif method == "volatility_parity":
            return VolatilityParitySizer(**params)
        else:
            raise ValueError(f"Unknown position sizer method: {method}")

    def determine_allocations(self,
                              candidate_analysis: List[Dict],
                              current_date: date,
                              total_portfolio_value: float,
                              historical_returns: Optional[pd.DataFrame] = None,
                              adv_data: Optional[Dict[str, float]] = None,
                              emergency_factor: Optional[float] = None) -> List[Dict[str, Any]]:
        self.logger.info("--- [Cognitive Engine Call: Marshal Coordination] ---")
        
        # --- Emergency Override ---
        if emergency_factor is not None:
            self.logger.warning(f"EMERGENCY FACTOR '{emergency_factor}' ACTIVATED. Overriding standard logic.")
            battle_plan = self.position_sizer.emergency_resize(emergency_factor)
            return battle_plan

        # [NEW] 1. Calculate the average cognitive uncertainty for the day from worthy targets
        worthy_targets = self.portfolio_constructor.identify_opportunities(candidate_analysis, current_date)
        
        daily_uncertainty = 0.0
        if worthy_targets and self.config.ai_mode != 'off':
            daily_asset_analysis = self.portfolio_constructor.asset_analysis_data.get(current_date, {})
            uncertainties = [daily_asset_analysis.get(t['ticker'], {}).get('final_conclusion', {}).get('posterior_variance', 0.0) for t in worthy_targets]
            valid_uncertainties = [u for u in uncertainties if u is not None]
            if valid_uncertainties:
                daily_uncertainty = sum(valid_uncertainties) / len(valid_uncertainties)

        # 2. Get the capital modifier based on this uncertainty
        capital_modifier = self.risk_manager.get_capital_modifier(daily_uncertainty)
        effective_max_allocation = self.config.max_total_allocation * capital_modifier
        
        # 3. [V2.0+] Use the dedicated position sizer to determine capital allocation
        # This correctly separates the "what" (worthy_targets) from the "how much" (sizer).
        initial_battle_plan = self.position_sizer.size_positions(worthy_targets, effective_max_allocation)

        # 4. [V2.0+] Apply liquidity constraints first
        battle_plan = self.risk_manager.apply_liquidity_constraints(
            initial_battle_plan, adv_data or {}, total_portfolio_value
        )

        # 5. [V2.0+] Enforce CVaR constraints on the liquidity-adjusted plan
        if self.risk_manager.risk_config.cvar_enabled and historical_returns is not None and battle_plan:
            portfolio_weights = {p['ticker']: p['capital_allocation_pct'] for p in battle_plan}
            portfolio_cvar = self.risk_manager.calculate_portfolio_cvar(portfolio_weights, historical_returns)

            if portfolio_cvar is not None and portfolio_cvar > self.risk_manager.risk_config.cvar_max_threshold:
                self.logger.warning(
                    f"Portfolio CVaR ({portfolio_cvar:.2%}) exceeds threshold "
                    f"({self.risk_manager.risk_config.cvar_max_threshold:.2%}). Scaling down."
                )
                # Simple scaling: reduce allocation proportionally to the CVaR excess
                scale_factor = self.risk_manager.risk_config.cvar_max_threshold / portfolio_cvar
                for position in battle_plan:
                    position['capital_allocation_pct'] *= scale_factor
                
                self.logger.info(f"Scaled down allocations by a factor of {scale_factor:.2f}.")

        self.logger.info("--- [Cognitive Engine Call: Concluded] ---")
        return battle_plan

    def calculate_opportunity_score(self, current_price: float, sma: float, rsi: float) -> float:
        """
        Calculates a proprietary 'opportunity score' based on technical indicators.
        Score is normalized to be between 0 and 100.
        """
        # Component 1: Trend (Price vs. SMA)
        # We want price to be above SMA, but not too far above.
        price_vs_sma = (current_price - sma) / sma
        # Use a gaussian-like function to reward being slightly above SMA
        trend_score = 100 * (1 - abs(price_vs_sma - 0.05)) # Peaking at 5% above SMA
        trend_score = max(0, trend_score) # Ensure non-negative

        # Component 2: Momentum/Mean-Reversion (RSI)
        # We want to buy when RSI is not overbought.
        rsi_score = 100 - rsi if rsi > self.config.rsi_overbought_threshold else 100
        
        # Combine the scores. Let's give more weight to the trend.
        final_score = (0.6 * trend_score) + (0.4 * rsi_score)
        return max(0, min(100, final_score))

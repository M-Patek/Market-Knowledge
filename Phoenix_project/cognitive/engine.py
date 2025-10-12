# cognitive/engine.py
import logging
from typing import List, Dict, Optional
from datetime import date

from phoenix_project import StrategyConfig, PositionSizerConfig
from sizing.base import IPositionSizer
from sizing.fixed_fraction import FixedFractionSizer
from .risk_manager import RiskManager
from .portfolio_constructor import PortfolioConstructor


class CognitiveEngine:
    def __init__(self, config: StrategyConfig, asset_analysis_data: Optional[Dict[date, Dict]] = None, sentiment_data: Optional[Dict[date, float]] = None, ai_mode: str = "processed"):
        self.config = config
        self.logger = logging.getLogger("PhoenixProject.CognitiveEngine")
        self.risk_manager = RiskManager(config, sentiment_data)
        self.portfolio_constructor = PortfolioConstructor(config, asset_analysis_data, mode=config.ai_mode)
        self.position_sizer = self._create_sizer(config.position_sizer)

    def _create_sizer(self, sizer_config: PositionSizerConfig) -> IPositionSizer:
        method = sizer_config.method
        params = sizer_config.parameters
        self.logger.info(f"Initializing position sizer: '{method}' with params: {params}")
        if method == "fixed_fraction":
            return FixedFractionSizer(**params)
        else:
            raise ValueError(f"Unknown position sizer method: {method}")

    def determine_allocations(self,
                              candidate_analysis: List[Dict],
                              current_vix: float,
                              current_date: date,
                              emergency_factor: Optional[float] = None) -> List[Dict]:
        self.logger.info("--- [Cognitive Engine Call: Marshal Coordination] ---")

        # --- Emergency Override Logic ---
        if emergency_factor is not None and emergency_factor < 1.0:
            self.logger.critical(f"EMERGENCY OVERRIDE: Injected factor of {emergency_factor:.3f} detected. "
                                 f"Overriding all models and issuing forced liquidation order.")
            # Create a battle plan to exit all positions. Ticker is a placeholder.
            battle_plan = [{"ticker": "ALL_ASSETS", "capital_allocation_pct": 0.0}]
            return battle_plan

        # --- Standard Logic ---
        capital_modifier = self.risk_manager.get_capital_modifier(current_vix, current_date)
        worthy_targets = self.portfolio_constructor.identify_opportunities(candidate_analysis, current_date)
        effective_max_allocation = self.config.max_total_allocation * capital_modifier
        battle_plan = self.position_sizer.size_positions(worthy_targets, effective_max_allocation)
        
        self.logger.info("--- [Cognitive Engine's Final Battle Plan] ---")
        final_total_allocation = sum(d['capital_allocation_pct'] for d in battle_plan)
        self.logger.info(f"Final planned capital deployment: {final_total_allocation:.2%}")
        for deployment in battle_plan: self.logger.info(f"- Asset: {deployment['ticker']}, Deploy Capital: {deployment['capital_allocation_pct']:.2%}")
        return battle_plan

    def calculate_opportunity_score(self, current_price: float, current_sma: float, current_rsi: float) -> float:
        return self.portfolio_constructor.calculate_opportunity_score(current_price, current_sma, current_rsi)

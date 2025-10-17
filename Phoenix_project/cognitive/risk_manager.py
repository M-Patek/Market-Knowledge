# cognitive/risk_manager.py
import logging
import numpy as np
import pandas as pd
from datetime import date
from typing import Optional, Dict, List, Any, Tuple

from phoenix_project import StrategyConfig

class RiskManager:
    """
    Manages overall portfolio risk by adjusting capital allocation based on the
    MetaLearner's cognitive uncertainty.
    """
    def __init__(self, config: StrategyConfig):
        """
        Initializes the RiskManager.
        Args:
            config: A dictionary containing the 'risk_manager' configuration.
        """
        self.config = config
        self.risk_config = self.config.risk_manager
        self.logger = logging.getLogger("PhoenixProject.RiskManager")
        self.logger.info(
            f"RiskManager initialized. "
            f"Uncertainty modifier enabled (max_uncertainty={self.risk_config.max_uncertainty}). "
            f"CVaR enabled ({self.risk_config.cvar_enabled}) with alpha={self.risk_config.cvar_alpha} "
            f"and threshold={self.risk_config.cvar_max_threshold}.")

    def get_capital_modifier(self, meta_learner_uncertainty: float) -> float:
        """
        Determines the capital allocation modifier based on the MetaLearner's
        posterior variance (uncertainty). Higher uncertainty leads to a lower modifier.
        """
        if meta_learner_uncertainty < 0:
            self.logger.warning(f"Received negative uncertainty ({meta_learner_uncertainty}). Clamping to 0.")
            meta_learner_uncertainty = 0

        if meta_learner_uncertainty >= self.risk_config.max_uncertainty or self.risk_config.max_uncertainty == 0:
            modifier = self.risk_config.min_capital_modifier
        else:
            # Linearly scale the modifier from 1.0 (at uncertainty 0) down to min_modifier (at max_uncertainty)
            slope = (self.risk_config.min_capital_modifier - 1.0) / self.risk_config.max_uncertainty
            modifier = 1.0 + slope * meta_learner_uncertainty

        final_modifier = max(self.risk_config.min_capital_modifier, min(modifier, 1.0))
        self.logger.info(f"MetaLearner uncertainty is {meta_learner_uncertainty:.4f}. Capital modifier set to {final_modifier:.2%}")
        return final_modifier

    def calculate_portfolio_cvar(self, portfolio_weights: Dict[str, float], historical_returns: pd.DataFrame) -> Optional[float]:
        """
        Calculates the historical Conditional Value-at-Risk (CVaR) for a given set of portfolio weights.

        Args:
            portfolio_weights: A dictionary mapping tickers to their proposed weights.
            historical_returns: A DataFrame where columns are ticker symbols and rows are daily returns.

        Returns:
            The calculated CVaR of the portfolio, or None if calculation fails.
        """
        if not self.risk_config.cvar_enabled:
            return 0.0

        if historical_returns.empty or not portfolio_weights:
            return 0.0

        tickers = list(portfolio_weights.keys())
        weights = np.array(list(portfolio_weights.values()))

        portfolio_returns = historical_returns[tickers].dot(weights)
        var = np.percentile(portfolio_returns, (1 - self.risk_config.cvar_alpha) * 100)
        cvar = portfolio_returns[portfolio_returns <= var].mean()

        return abs(cvar) if pd.notna(cvar) else None

    def apply_liquidity_constraints(self, battle_plan: List[Dict[str, Any]], adv_data: Dict[str, float], total_portfolio_value: float) -> List[Dict[str, Any]]:
        """
        Adjusts a battle plan based on liquidity constraints (ADV).

        Args:
            battle_plan: The proposed portfolio allocations.
            adv_data: A dictionary mapping tickers to their Average Daily Volume in dollars.
            total_portfolio_value: The total current value of the portfolio.

        Returns:
            A revised battle plan with allocations capped by liquidity constraints.
        """
        if not self.risk_config.liquidity_management_enabled or not battle_plan:
            return battle_plan

        adjusted_plan = []
        for position in battle_plan:
            ticker = position['ticker']
            adv = adv_data.get(ticker)

            if adv is None or adv == 0:
                self.logger.warning(f"No ADV data for '{ticker}'. Excluding from plan due to liquidity risk.")
                continue

            max_position_value = adv * self.risk_config.max_adv_participation_rate
            max_allocation_pct = max_position_value / total_portfolio_value

            if position['capital_allocation_pct'] > max_allocation_pct:
                self.logger.info(f"Liquidity constraint hit for '{ticker}'. Capping allocation from {position['capital_allocation_pct']:.2%} to {max_allocation_pct:.2%}.")
                position['capital_allocation_pct'] = max_allocation_pct
            
            adjusted_plan.append(position)

        return adjusted_plan

# cognitive/risk_manager.py
import logging
from typing import Optional, Dict
from datetime import date

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
        self.logger.info(f"RiskManager initialized for Dynamic Risk Budgeting with max_uncertainty={self.risk_config.max_uncertainty} and min_modifier={self.risk_config.min_capital_modifier}")

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

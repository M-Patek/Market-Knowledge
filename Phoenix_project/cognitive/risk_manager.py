# cognitive/risk_manager.py
import logging
from typing import Optional, Dict
from datetime import date

from phoenix_project import StrategyConfig

class RiskManager:
    def __init__(self, config: StrategyConfig, sentiment_data: Optional[Dict[date, float]] = None):
        self.config = config
        self.sentiment_data = sentiment_data if sentiment_data is not None else {}
        self.logger = logging.getLogger("PhoenixProject.RiskManager")
        if self.sentiment_data: self.logger.info(f"RiskManager initialized with {len(self.sentiment_data)} days of sentiment data.")

    def get_capital_modifier(self, current_vix: float, current_date: date) -> float:
        self.logger.info(f"Assessing risk for {current_date.isoformat()}. VIX: {current_vix:.2f}")
        if current_vix > self.config.vix_high_threshold:
            base_modifier = self.config.capital_modifier_high_vix
            self.logger.info(f"VIX indicates High Fear. Base modifier: {base_modifier:.2f}")
        elif current_vix < self.config.vix_low_threshold:
            base_modifier = self.config.capital_modifier_low_vix
            self.logger.info(f"VIX indicates Low Fear. Base modifier: {base_modifier:.2f}")
        else:
            base_modifier = self.config.capital_modifier_normal_vix
            self.logger.info(f"VIX indicates Normal Fear. Base modifier: {base_modifier:.2f}")
        if not self.sentiment_data: return base_modifier
        sentiment_score = self.sentiment_data.get(current_date, 0.0)
        sentiment_adjustment = 1.0 + (sentiment_score * 0.2)
        final_modifier = base_modifier * sentiment_adjustment
        final_modifier = max(0.0, min(1.1, final_modifier))
        self.logger.info(f"Gemini Sentiment Score: {sentiment_score:.2f}. Final Capital Modifier: {final_modifier:.2%}")
        return final_modifier

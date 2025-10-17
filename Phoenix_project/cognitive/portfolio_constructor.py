# cognitive/portfolio_constructor.py
import logging
from typing import List, Dict, Optional, Any
from datetime import date
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from pydantic import BaseModel
from phoenix_project import StrategyConfig

class PortfolioConstructor:
    """
    Transforms raw model predictions into a structured, actionable portfolio state.
    """
    def __init__(self, config: StrategyConfig, asset_analysis_data: Optional[Dict[date, Dict]] = None, mode: str = "processed"):
        """
        Initializes the PortfolioConstructor with strategy configurations.
        Args:
            config: A dictionary containing the 'portfolio_constructor' configuration.
        """
        self.logger = logging.getLogger("PhoenixProject.PortfolioConstructor")
        self.config = config
        self.asset_analysis_data = asset_analysis_data if asset_analysis_data is not None else {}
        self.mode = mode
        self.portfolio_state = {}
        self.score_weights = self.config.portfolio_constructor.score_weights
        self.ema_span = self.config.portfolio_constructor.ema_span
        # [V2.0+] Get optimizer config
        self.optimizer_config = self.config.portfolio_optimizer
        self.logger.info(
            f"PortfolioConstructor initialized. Optimizer method: '{self.optimizer_config.method}'. "
            f"Weights={self.score_weights} and ema_span={self.ema_span}."
        )

    def identify_opportunities(self, candidate_analysis: List[Dict], current_date: date) -> List[Dict]:
        self.logger.info("Identifying opportunities from daily candidate analysis...")
        if not candidate_analysis:
            self.logger.warning("No candidate analysis provided.")
            return []

        if self.mode != 'off' and not self.asset_analysis_data:
            self.logger.error("AI mode is enabled but no pre-computed asset analysis data was provided.")
            return []

        daily_asset_analysis = self.asset_analysis_data.get(current_date, {}) if self.mode != 'off' else {}
        
        worthy_targets = []
        for candidate in candidate_analysis:
            ticker = candidate['ticker']
            opportunity_score = candidate.get('opportunity_score', 0.0)
            
            final_conclusion = daily_asset_analysis.get(ticker, {}).get('final_conclusion', {})
            confidence_score = final_conclusion.get('final_probability', 0.5) if final_conclusion else 0.5
            
            # Combine the scores
            final_score = self._calculate_final_score(opportunity_score, confidence_score)
            
            if final_score >= self.config.opportunity_score_threshold:
                self.logger.info(f"'{ticker}' identified as a worthy target. Final Score: {final_score:.2f}, Opportunity Score: {opportunity_score:.2f}, AI Confidence: {confidence_score:.2f}")
                worthy_targets.append({
                    "ticker": ticker,
                    "final_score": final_score,
                    "volatility": candidate.get("volatility")
                })
        
        # Sort by score to prioritize the best opportunities
        worthy_targets.sort(key=lambda x: x['final_score'], reverse=True)
        self.logger.info(f"Identified {len(worthy_targets)} worthy targets for {current_date.isoformat()}.")
        return worthy_targets

    def _calculate_final_score(self, alpha_score: float, confidence_score: float) -> float:
        """
        Calculates the final, blended score for an asset.
        """
        # Weighted average of alpha and confidence
        alpha_weight = self.score_weights.get("alpha_score", 0.5)
        confidence_weight = self.score_weights.get("confidence_score", 0.5)
        
        final_score = (alpha_score * alpha_weight) + (confidence_score * confidence_weight)
        return final_score

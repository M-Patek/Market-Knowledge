# cognitive/portfolio_constructor.py
import logging
from typing import List, Dict
from datetime import date

from phoenix_project import StrategyConfig

class PortfolioConstructor:
    def __init__(self, config: StrategyConfig, asset_analysis_data: Dict[date, Dict] | None = None, mode: str = "processed"):
        self.config = config
        self.asset_analysis_data = asset_analysis_data or {}
        self.logger = logging.getLogger("PhoenixProject.PortfolioConstructor")
        self.mode = mode
        self._ema_state = {}
        self.ema_alpha = 0.2
        self.global_scale = 1.0

    def calculate_opportunity_score(self, current_price: float, current_sma: float, current_rsi: float) -> float:
        if current_sma <= 0: return 0.0
        momentum_score = 50 + 50 * ((current_price / current_sma) - 1)
        if momentum_score > 50 and current_rsi > self.config.rsi_overbought_threshold:
            overbought_intensity = (current_rsi - self.config.rsi_overbought_threshold) / (100 - self.config.rsi_overbought_threshold)
            penalty_factor = 1.0 - (overbought_intensity * 0.5)
            final_score = momentum_score * penalty_factor
        else:
            final_score = momentum_score
        return max(0.0, min(100.0, final_score))
        
    def _sanitize_ai_output(self, raw: Dict) -> tuple[float, float]:
        try:
            f = float(raw.get("adjustment_factor", 1.0))
            c = float(raw.get("confidence", 0.0))
        except (ValueError, TypeError): return 1.0, 0.0
        f = max(0.3, min(2.0, f))
        c = max(0.0, min(1.0, c))
        return f, c

    def _effective_factor(self, ticker: str, reported_factor: float, confidence: float) -> float:
        effective = 1.0 + confidence * (reported_factor - 1.0)
        prev = self._ema_state.get(ticker, 1.0)
        smoothed = prev * (1 - self.ema_alpha) + effective * self.ema_alpha
        self._ema_state[ticker] = smoothed
        final = smoothed * self.global_scale
        final = max(0.5, min(1.2, final))
        return final

    def identify_opportunities(self, candidate_analysis: List[Dict], current_date: date) -> List[Dict]:
        self.logger.info("PortfolioConstructor is identifying high-quality opportunities...")
        adjusted_candidates = []
        daily_asset_analysis = self.asset_analysis_data.get(current_date, {})
        
        for candidate in candidate_analysis:
            ticker = candidate["ticker"]
            original_score = candidate["opportunity_score"]
            final_factor = 1.0
            confidence = 0.0
            
            if self.mode != 'off':
                fused_response_data = daily_asset_analysis.get(ticker)
                if fused_response_data:
                    final_factor = fused_response_data.get('final_factor', 1.0)
                    confidence = 1.0 - fused_response_data.get('dispersion', 1.0) # Use dispersion as an inverse proxy for confidence
            adjusted_score = candidate["opportunity_score"] * final_factor
            adjusted_candidates.append({**candidate, "adjusted_score": adjusted_score, "ai_factor": final_factor, "ai_confidence": confidence})
            if final_factor != 1.0 and self.mode != 'off': self.logger.info(f"AI Insight for {ticker} (Mode: {self.mode}): Conf={confidence:.2f}, FinalFactor={final_factor:.3f}. Score: {original_score:.2f} -> {adjusted_score:.2f}")
        
        worthy_targets = [res for res in adjusted_candidates if res["adjusted_score"] > self.config.opportunity_score_threshold]
        if not worthy_targets:
            self.logger.info("PortfolioConstructor: No opportunities met the threshold.")
        else:
            self.logger.info(f"PortfolioConstructor: Identified {len(worthy_targets)} high-quality opportunities.")
        return worthy_targets

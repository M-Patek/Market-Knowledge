# strategy_handler.py
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
import backtrader as bt

from phoenix_project import StrategyConfig

@dataclass
class DailyDataPacket:
    """A simple container for all data needed by the strategy for a single day."""
    current_date: Any
    current_vix: Optional[float]
    current_yield: Optional[float]
    current_breadth: Optional[float]
    candidate_analysis: list

class StrategyDataHandler:
    """
    Manages all data feeds, indicators, and external data lookups for the strategy.
    Its purpose is to decouple data management from the core strategy decision-making logic.
    """
    def __init__(self, strategy: bt.Strategy, config: StrategyConfig, vix_data: pd.Series, treasury_yield_data: pd.Series, market_breadth_data: pd.Series):
        self.strategy = strategy
        self.config = config
        self.logger = logging.getLogger("PhoenixProject.StrategyDataHandler")
        
        # Data and indicator setup
        self.data_map = {d._name: d for d in self.strategy.datas}
        self.sma_indicators = {
            d._name: bt.indicators.SimpleMovingAverage(d.close, period=self.config.sma_period)
            for d in self.strategy.datas
        }
        self.rsi_indicators = {
            d._name: bt.indicators.RSI(d.close, period=self.config.rsi_period)
            for d in self.strategy.datas
        }
        
        # External data lookups
        self.vix_lookup = {pd.Timestamp(k).date(): float(v) for k, v in vix_data.items()}
        self.yield_lookup = {pd.Timestamp(k).date(): float(v) for k, v in treasury_yield_data.items()}
        self.breadth_lookup = {pd.Timestamp(k).date(): float(v) for k, v in market_breadth_data.items()}
        
        self.logger.info("StrategyDataHandler initialized and has prepared all indicators.")

    def get_daily_data_packet(self, cognitive_engine) -> Optional[DailyDataPacket]:
        """
        Prepares and returns a data packet for the current day if all data is valid.
        """
        current_date = self.strategy.datas[0].datetime.date(0)
        
        current_vix = self.vix_lookup.get(current_date)
        if current_vix is None:
            self.logger.warning(f"Critical data VIX missing for {current_date}, halting for the day.")
            return None
            
        current_yield = self.yield_lookup.get(current_date)
        current_breadth = self.breadth_lookup.get(current_date)
        
        self.logger.info(f"VIX Index: {current_vix:.2f}, 10Y Yield: {current_yield:.2f if current_yield else 'N/A'}%, Market Breadth: {current_breadth:.2% if current_breadth else 'N/A'}")
        
        candidate_analysis = [{
            "ticker": ticker,
            "opportunity_score": cognitive_engine.calculate_opportunity_score(
                d.close[0], self.sma_indicators[ticker][0], self.rsi_indicators[ticker][0]
            )
        } for ticker, d in self.data_map.items()]
        
        return DailyDataPacket(
            current_date=current_date,
            current_vix=current_vix,
            current_yield=current_yield,
            current_breadth=current_breadth,
            candidate_analysis=candidate_analysis
        )

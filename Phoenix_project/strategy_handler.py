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

        self.data_map = {d._name: d for d in self.strategy.datas}
        self.vix_data = vix_data
        self.treasury_yield_data = treasury_yield_data
        self.market_breadth_data = market_breadth_data

        # Initialize all indicators in a structured way
        self.indicators = self._initialize_indicators()

        self.logger.info("StrategyDataHandler initialized and has prepared all indicators.")

    def _initialize_indicators(self) -> Dict[str, Dict[str, Any]]:
        """Creates and organizes all indicators for each data feed."""
        indicators = {}
        for name, data in self.data_map.items():
            indicators[name] = {
                'sma': bt.indicators.SimpleMovingAverage(
                    data.close, period=self.config.sma_period
                ),
                'rsi': bt.indicators.RSI(
                    data.close, period=self.config.rsi_period
                )
            }
        return indicators

    def get_daily_data_packet(self, cognitive_engine) -> Optional[DailyDataPacket]:
        """
        Prepares and returns a data packet for the current day if all data is valid.
        """
        current_date = self.strategy.datas[0].datetime.date(0)

        try:
            # Use direct, efficient lookup on the Series
            current_vix = self.vix_data.loc[pd.Timestamp(current_date)]
            current_yield = self.treasury_yield_data.loc[pd.Timestamp(current_date)]
            current_breadth = self.market_breadth_data.loc[pd.Timestamp(current_date)]
        except KeyError:
            self.logger.warning(f"Critical data VIX missing for {current_date}, halting for the day.")
            return None

        self.logger.info(f"VIX Index: {current_vix:.2f}, 10Y Yield: {current_yield:.2f if current_yield else 'N/A'}%, Market Breadth: {current_breadth:.2% if current_breadth else 'N/A'}")

        candidate_analysis = [{
            "ticker": ticker,
            "opportunity_score": cognitive_engine.calculate_opportunity_score(
                d.close[0],
                self.indicators[ticker]['sma'][0],
                self.indicators[ticker]['rsi'][0]
            )
        } for ticker, d in self.data_map.items()]

        return DailyDataPacket(
            current_date=current_date,
            current_vix=current_vix,
            current_yield=current_yield,
            current_breadth=current_breadth,
            candidate_analysis=candidate_analysis
        )

# strategy_handler.py
import logging
from ai.market_state_predictor import MarketStatePredictor
from typing import Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
import backtrader as bt
from datetime import date

from phoenix_project import StrategyConfig
from cognitive.engine import CognitiveEngine
from features.store import SimpleFeatureStore # [V2.0] Import the feature store
from execution.order_manager import OrderManager

@dataclass
class DailyDataPacket:
    current_date: date
    current_vix: Optional[float]
    current_yield: Optional[float]
    current_breadth: Optional[float]
    market_state: Optional[int]
    market_state_confidence: Optional[float]
    # Each dict will now include 'volatility'
    candidate_analysis: list

class StrategyDataHandler:
    """
    Acts as a data provider and pre-processor for the main strategy logic.
    Its purpose is to decouple data management from the core strategy decision-making logic.
    """
    def __init__(self, strategy: bt.Strategy, config: StrategyConfig, vix_data: pd.Series, treasury_yield_data: pd.Series, market_breadth_data: pd.Series, feature_store: SimpleFeatureStore, market_state_predictor: Optional[MarketStatePredictor] = None):
        self.strategy = strategy
        self.config = config
        self.logger = logging.getLogger("PhoenixProject.StrategyDataHandler")
        self.market_state_predictor = market_state_predictor
        self.feature_store = feature_store # [V2.0] Use the feature store

        self.data_map = {d._name: d for d in self.strategy.datas}
        self.vix_data = vix_data
        self.treasury_yield_data = treasury_yield_data
        self.market_breadth_data = market_breadth_data

    def get_daily_data_packet(self, cognitive_engine: CognitiveEngine) -> Optional[DailyDataPacket]:
        """Assembles all necessary data for the current day into a single packet."""
        try:
            current_date = self.strategy.datetime.date()
            current_timestamp = pd.Timestamp(current_date)
        except IndexError:
            self.logger.warning("Could not get current date from strategy datetime.")
            return None

        # Fetch macro data
        current_vix = self.vix_data.loc[current_timestamp]
        current_yield = self.treasury_yield_data.loc[current_timestamp]
        current_breadth = self.market_breadth_data.loc[current_timestamp]
        self.logger.info(f"VIX Index: {current_vix:.2f}, 10Y Yield: {current_yield:.2f if current_yield else 'N/A'}%, Market Breadth: {current_breadth:.2% if current_breadth else 'N/A'}")

        market_state, market_state_confidence = 1, 0.0 # Default to neutral
        if self.market_state_predictor:
            macro_features = pd.DataFrame([{
                'vix': current_vix,
                'yield': current_yield,
                'breadth': current_breadth
            }])
            market_state, market_state_confidence = self.market_state_predictor.predict(macro_features)

        candidate_analysis = []
        for ticker, d in self.data_map.items():
            # [V2.0] Get features from the centralized feature store
            # We convert the backtrader data lines to a pandas DataFrame
            df = pd.DataFrame({'close': d.close.get(size=self.config.sma_period + 1)})
            features = self.feature_store.get_features(ticker, df)
            
            candidate_analysis.append({
                "ticker": ticker,
                "opportunity_score": cognitive_engine.calculate_opportunity_score(
                    d.close[0], features.get('sma'), features.get('rsi')
                ),
                "volatility": features.get('volatility')
            })

        return DailyDataPacket(
            current_date=current_date,
            current_vix=current_vix,
            current_yield=current_yield,
            current_breadth=current_breadth,
            market_state=market_state,
            market_state_confidence=market_state_confidence,
            candidate_analysis=candidate_analysis
        )


class RomanLegionStrategy(bt.Strategy):
    def __init__(self, config: StrategyConfig, vix_data, treasury_yield_data, market_breadth_data, sentiment_data, asset_analysis_data, market_state_predictor: Optional[MarketStatePredictor] = None):
        self.config = config
        self.logger = logging.getLogger("PhoenixProject.RomanLegionStrategy")
        # [V2.0] Initialize the feature store first
        self.feature_store = SimpleFeatureStore(config.dict())
        self.cognitive_engine = CognitiveEngine(config, asset_analysis_data, sentiment_data)
        self.order_manager = OrderManager(self.broker, **config.execution_model.dict())
        # [V2.0] Pass the feature_store to the data handler
        self.data_handler = StrategyDataHandler(self, config, vix_data, treasury_yield_data, market_breadth_data, self.feature_store, market_state_predictor)
        
    def next(self):
        daily_data = self.data_handler.get_daily_data_packet(self.cognitive_engine)
        if not daily_data:
            return

        self.logger.info(f"--- {daily_data.current_date.isoformat()}: Daily Rebalancing Briefing ---")
        battle_plan = self.cognitive_engine.determine_allocations(daily_data.candidate_analysis, daily_data.current_date)
        self.order_manager.rebalance(self, battle_plan)

    def notify_order(self, order):
        self.order_manager.handle_order_notification(order)

    def stop(self):
        self.logger.info("--- [Strategy Stop]: Finalizing operations ---")
        # Any final logic can go here
        pass


# features/store.py
import backtrader as bt
from typing import Dict, Any
import pandas as pd
from .base import IFeatureStore

class SimpleFeatureStore(IFeatureStore):
    """
    A simple, in-memory feature store that calculates technical indicators.
    This centralizes the feature generation logic.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def get_features(self, ticker: str, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates SMA, RSI, and Volatility for a given dataset.
        In a real system, this could fetch pre-calculated features from a database.
        """
        # Note: This is a simplified calculation for demonstration.
        # It doesn't use backtrader indicators directly to be framework-agnostic.
        close_series = data['close']
        
        sma = close_series.rolling(window=self.config.get('sma_period', 50)).mean().iloc[-1]
        
        delta = close_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.config.get('rsi_period', 14)).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config.get('rsi_period', 14)).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]

        volatility = close_series.rolling(
            window=self.config.get('position_sizer', {}).get('parameters', {}).get('volatility_period', 20)
        ).std().iloc[-1]

        return {
            'sma': sma,
            'rsi': rsi,
            'volatility': volatility
        }


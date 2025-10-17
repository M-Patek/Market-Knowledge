# features/store.py
from typing import Dict, Any
from datetime import date
import pandas as pd
from .base import IFeatureStore

class _BaseFeatureCalculator:
    """
    Houses the core feature calculation logic to be shared across
    online and offline stores, ensuring no skew.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def _calculate_features(self, pit_data: pd.DataFrame) -> Dict[str, Any]:
        if pit_data.empty or len(pit_data) < self.config.get('sma_period', 50):
            return {'sma': None, 'rsi': None, 'volatility': None}

        close_series = pit_data['close']

        sma = close_series.rolling(window=self.config.get('sma_period', 50)).mean().iloc[-1]

        delta = close_series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.config.get('rsi_period', 14)).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config.get('rsi_period', 14)).mean()
        
        # Avoid division by zero for RSI
        if loss.iloc[-1] == 0:
            rsi = 100.0
        else:
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]

        volatility_period = self.config.get('position_sizer', {}).get('parameters', {}).get('volatility_period', 20)
        volatility = close_series.rolling(window=volatility_period).std().iloc[-1]

        return {
            'sma': sma,
            'rsi': rsi,
            'volatility': volatility
        }

class OfflineFeatureStore(_BaseFeatureCalculator, IFeatureStore):
    """
    Feature store for offline use (e.g., backtesting, training).
    It operates on a historical DataFrame passed in memory.
    """
    def get_features(self, ticker: str, as_of_date: date, full_history: pd.DataFrame) -> Dict[str, Any]:
        # Ensure point-in-time correctness by filtering the DataFrame
        # We use '<' to ensure the data from 'as_of_date' itself is not included.
        pit_data = full_history[full_history.index < pd.Timestamp(as_of_date)]
        return self._calculate_features(pit_data)

class OnlineFeatureStore(_BaseFeatureCalculator, IFeatureStore):
    """
    Feature store for online use (e.g., real-time prediction).
    Connects to low-latency sources like Redis or Kafka.
    """
    def __init__(self, config: Dict[str, Any], redis_client=None):
        """
        In a real implementation, this would establish connections to
        real-time data stores.
        """
        super().__init__(config)
        self.redis_client = redis_client # Placeholder for a real client

    def get_features(self, ticker: str, as_of_date: date, full_history: pd.DataFrame = None) -> Dict[str, Any]:
        """
        For online serving, 'full_history' is ignored. Instead, it would
        fetch the last N data points from a real-time store (e.g., Redis)
        to calculate features with low latency.
        """
        # TODO: Implement actual real-time data fetching logic.
        # 1. Connect to Redis/Kafka.
        # 2. Fetch the last ~252 trading days of data for the ticker.
        # 3. Construct a pandas DataFrame `pit_data`.
        # 4. Return self._calculate_features(pit_data)
        raise NotImplementedError(
            "OnlineFeatureStore.get_features is not yet implemented. "
            "It must be connected to a real-time data source."
        )

# ai/market_state_predictor.py
import logging
import pandas as pd
import numpy as np
from typing import Tuple
import lightgbm as lgb

def generate_market_state_labels(market_data: pd.DataFrame, long_term_ma: int = 200, threshold: float = 0.05) -> pd.Series:
    """
    Generates market state labels (0: Bear, 1: Sideways, 2: Bull) based on a long-term moving average.
    """
    ma = market_data['Close'].rolling(window=long_term_ma).mean()
    upper_band = ma * (1 + threshold)
    lower_band = ma * (1 - threshold)

    conditions = [
        market_data['Close'] < lower_band,
        market_data['Close'] > upper_band
    ]
    choices = [0, 2] # 0 for Bear, 2 for Bull
    labels = np.select(conditions, choices, default=1) # 1 for Sideways
    return pd.Series(labels, index=market_data.index, name="market_state")

class MarketStatePredictor:
    """
    A base learner to predict the market state from macro-economic features.
    """
    def __init__(self, config: dict):
        self.logger = logging.getLogger("PhoenixProject.MarketStatePredictor")
        self.config = config
        self.model = lgb.LGBMClassifier(**self.config.get('model_params', {}))
        self.is_trained = False

    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Trains the market state classifier.
        Args:
            X: DataFrame of macro features (e.g., VIX, yield, breadth).
            y: Series of market state labels (0, 1, 2).
        """
        self.logger.info(f"Training MarketStatePredictor on {len(X)} data points...")
        self.model.fit(X, y)
        self.is_trained = True
        self.logger.info("MarketStatePredictor training complete.")

    def predict(self, X_live: pd.DataFrame) -> Tuple[int, float]:
        """
        Predicts the market state and confidence for live data.
        Returns:
            A tuple of (predicted_state, confidence_score).
            Confidence is the probability of the predicted class.
        """
        if not self.is_trained:
            self.logger.warning("Predict called before training. Returning neutral state.")
            return 1, 0.0 # Neutral state (Sideways) with zero confidence
        
        probabilities = self.model.predict_proba(X_live)
        predicted_class_index = np.argmax(probabilities, axis=1)[0]
        confidence = probabilities[0][predicted_class_index]
        
        # We can also return a single "confidence" score (e.g., Bull prob - Bear prob)
        # For now, we return the probability of the *most likely* state.
        return predicted_class_index, confidence

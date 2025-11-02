# sizing/volatility_parity.py
import logging
import numpy as np
from typing import List, Dict, Any

# 修复：将相对导入 'from .base...' 更改为绝对导入
from sizing.base import IPositionSizer

class VolatilityParitySizer(IPositionSizer):
    """
    Sizes positions based on the inverse of their volatility, ensuring each
    asset contributes equally to the portfolio's overall risk.
    """
    def __init__(self, volatility_period: int):
        self.logger = logging.getLogger("PhoenixProject.VolatilityParitySizer")
        if volatility_period <= 1:
            raise ValueError("Volatility period must be greater than 1.")
        self.volatility_period = volatility_period
        self.logger.info(f"VolatilityParitySizer initialized with period: {self.volatility_period}")

    def size_positions(self, candidates: List[Dict[str, Any]], max_total_allocation: float) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        # Filter out candidates with zero or None volatility
        valid_candidates = [c for c in candidates if c.get('volatility') is not None and c['volatility'] > 0]
        if not valid_candidates:
            self.logger.warning("No candidates with valid volatility found. Cannot size positions.")
            return []

        inverse_volatility = [1 / c['volatility'] for c in valid_candidates]
        total_inverse_vol = sum(inverse_volatility)
        
        # Normalize the weights so they sum to the max total allocation
        risk_weights = [(iv / total_inverse_vol) * max_total_allocation for iv in inverse_volatility]

        battle_plan = [{"ticker": c['ticker'], "capital_allocation_pct": w} for c, w in zip(valid_candidates, risk_weights)]
        return battle_plan

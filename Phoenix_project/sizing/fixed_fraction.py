# sizing/fixed_fraction.py

import logging
from typing import List, Dict, Any

# 修复：将相对导入 'from .base...' 更改为绝对导入
from Phoenix_project.sizing.base import IPositionSizer


class FixedFractionSizer(IPositionSizer):
    """
    A simple position sizer that allocates a fixed fraction of the portfolio
    to each qualifying candidate asset.
    """
    def __init__(self, fraction_per_position: float):
        self.logger = logging.getLogger("PhoenixProject.FixedFractionSizer")
        if not (0 < fraction_per_position <= 1.0):
            raise ValueError("Fraction per position must be between 0 and 1.")
        self.fraction = fraction_per_position
        self.logger.info(f"FixedFractionSizer initialized with fraction: {self.fraction:.2%}")

    def size_positions(self, candidates: List[Dict[str, Any]], max_total_allocation: float) -> List[Dict[str, Any]]:
        if not candidates:
            return []

        self.logger.info(f"Sizing {len(candidates)} candidates with a fixed fraction of {self.fraction:.2%}")
        
        battle_plan = [
            {"ticker": candidate['ticker'], "capital_allocation_pct": self.fraction}
            for candidate in candidates
        ]

        # Enforce the global maximum allocation constraint
        total_planned_allocation = sum(p['capital_allocation_pct'] for p in battle_plan)
        if total_planned_allocation > max_total_allocation:
            self.logger.warning(
                f"Total fixed allocation {total_planned_allocation:.2%} exceeds cap of "
                f"{max_total_allocation:.2%}. Scaling down all positions."
            )
            scale_factor = max_total_allocation / total_planned_allocation
            for deployment in battle_plan:
                deployment['capital_allocation_pct'] *= scale_factor

        return battle_plan

# sizing/base.py

from typing import Protocol, List, Dict, Any


class IPositionSizer(Protocol):
    """
    Defines the interface for a position sizing strategy.
    """

    def size_positions(self, candidates: List[Dict[str, Any]], max_total_allocation: float) -> List[Dict[str, Any]]:
        """Calculates the capital allocation percentage for a list of trade candidates."""
        ...

# execution/interfaces.py
"""
Defines the abstract contracts (interfaces) for the execution layer.
This ensures a clean separation between the strategy's intent and the
mechanics of order submission and management.
"""
import logging
from typing import Protocol, List, Dict, Any, Literal
from dataclasses import dataclass
from datetime import datetime
import backtrader as bt


@dataclass
class Order:
    """A universal representation of a trading order."""
    ticker: str
    side: Literal['BUY', 'SELL']
    size: float
    order_type: Literal['MARKET', 'LIMIT'] = 'LIMIT'
    limit_price: float = 0.0
    created_at: datetime = datetime.utcnow()
    status: str = 'NEW'
    order_id: str = ''


class IBrokerAdapter(Protocol):
    """Interface for a connection to a broker or a simulation thereof."""

    def place_order(self, strategy: bt.Strategy, order: Order) -> Order:
        """Submits an order to the broker and returns the updated order with a broker-assigned ID."""
        ...

    def get_portfolio_value(self) -> float:
        """Returns the total current value of the portfolio."""
        ...


class IOrderManager(Protocol):
    """Interface for the central order management logic."""

    def rebalance(self, strategy: bt.Strategy, target_portfolio: List[Dict[str, Any]]) -> None:
        """
        Calculates and executes the necessary trades to align the current
        portfolio with the target portfolio.
        """
        ...

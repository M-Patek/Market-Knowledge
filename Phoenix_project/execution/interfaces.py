# execution/interfaces.py
"""
Defines the abstract contracts (interfaces) for the execution layer.
This ensures a clean separation between the strategy's intent and the
mechanics of order submission and management.
"""
import logging
from typing import Protocol, List, Dict, Any, Literal
from dataclasses import dataclass, field
from datetime import datetime
import backtrader as bt
import uuid

@dataclass
class Order:
    """A universal representation of a trading order."""
    ticker: str
    side: Literal['BUY', 'SELL']
    size: float
    order_type: Literal['Market', 'Limit'] = 'Limit'
    limit_price: float = 0.0
    fill_probability: float = 1.0 # The base probability of fill, calculated by the OrderManager
    created_at: datetime = field(default_factory=datetime.utcnow)
    status: str = 'NEW'
    order_id: str = field(default_factory=lambda: f"ord_{uuid.uuid4()}") # 确保 Order 有 ID

# --- 新增：补全缺失的 Fill Pydantic 模型 ---
# 供 drl/trading_env.py 和 execution/adapters.py 使用
@dataclass
class Fill:
    """Represents an executed trade (a fill)."""
    order_id: str
    symbol: str
    fill_amount: float
    fill_price: float
    timestamp: datetime
    fill_id: str = field(default_factory=lambda: f"fill_{uuid.uuid4()}")


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

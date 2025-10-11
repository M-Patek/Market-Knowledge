# execution/adapters.py
"""
Concrete implementations of the IBrokerAdapter interface.
This file will contain adapters for backtesting, paper trading, and live brokers.
"""
import logging
import backtrader as bt
from .interfaces import Order

class BacktraderBrokerAdapter:
    """
    An adapter that translates universal Order objects into backtrader
    buy/sell commands, allowing the OrderManager to be used within a backtest.
    """
    def __init__(self, broker: bt.Broker):
        self.logger = logging.getLogger("PhoenixProject.BacktraderAdapter")
        self.broker = broker
        self.logger.info("BacktraderBrokerAdapter initialized.")

    def place_order(self, strategy: bt.Strategy, order: Order) -> Order:
        """Places an order using the backtrader engine."""
        data = strategy.getdatabyname(order.ticker)
        if order.side == 'BUY':
            if order.limit_price < data.low[0]:
                self.logger.warning(f"BUY order for {order.ticker} likely unfilled. Limit Price ${order.limit_price:.2f} is below bar low ${data.low[0]:.2f}.")
                order.status = 'REJECTED'
                return order
            strategy.buy(data=data, size=order.size, price=order.limit_price, exectype=bt.Order.Limit)
            self.logger.info(f"Placed BUY order for {order.ticker}: Size={order.size:.2f}, Est. Limit Price=${order.limit_price:.2f}")
        
        elif order.side == 'SELL':
            if order.limit_price > data.high[0]:
                self.logger.warning(f"SELL order for {order.ticker} likely unfilled. Limit Price ${order.limit_price:.2f} is above bar high ${data.high[0]:.2f}.")
                order.status = 'REJECTED'
                return order
            strategy.sell(data=data, size=order.size, price=order.limit_price, exectype=bt.Order.Limit)
            self.logger.info(f"Placed SELL order for {order.ticker}: Size={order.size:.2f}, Est. Limit Price=${order.limit_price:.2f}")
        
        order.status = 'SUBMITTED'
        return order

    def get_portfolio_value(self) -> float:
        """Returns the total portfolio value from the backtrader broker."""
        return self.broker.getvalue()

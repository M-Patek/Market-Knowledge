# execution/adapters.py
"""
Concrete implementations of the IBrokerAdapter interface.
This file will contain adapters for backtesting, paper trading, and live brokers.
"""
import logging
import random
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
        if order.order_type == 'Market':
            if order.side == 'BUY':
                strategy.buy(data=strategy.getdatabyname(order.ticker), size=order.size)
            else:
                strategy.sell(data=strategy.getdatabyname(order.ticker), size=order.size)
            order.status = 'SUBMITTED'
        elif order.order_type == 'Limit':
            data = strategy.getdatabyname(order.ticker)
            next_bar_open = data.open[1]

            # [V2.0+] Adverse Selection Model
            adverse_selection_penalty = 0.0
            if next_bar_open > 0: # Avoid division by zero
                if order.side == 'BUY' and order.limit_price < next_bar_open:
                    # Penalize if our buy limit is "too good" (far below the open)
                    adverse_selection_penalty = (next_bar_open - order.limit_price) / next_bar_open
                elif order.side == 'SELL' and order.limit_price > next_bar_open:
                    # Penalize if our sell limit is "too good" (far above the open)
                    adverse_selection_penalty = (order.limit_price - next_bar_open) / next_bar_open

            final_fill_prob = order.fill_probability * (1.0 - adverse_selection_penalty)

            if random.random() < final_fill_prob:
                if order.side == 'BUY':
                    strategy.buy(data=data, size=order.size, price=order.limit_price, exectype=bt.Order.Limit)
                else: # SELL
                    strategy.sell(data=data, size=order.size, price=order.limit_price, exectype=bt.Order.Limit)
                order.status = 'SUBMITTED'
            else:
                self.logger.warning(f"LIMIT order for {order.ticker} failed simulation (Final Prob: {final_fill_prob:.2f}). Base Prob: {order.fill_probability:.2f}, Adverse Selection Penalty: {adverse_selection_penalty:.2f}")
                order.status = 'REJECTED'
        return order

    def get_portfolio_value(self) -> float:
        """Returns the total portfolio value from the backtrader broker."""
        return self.broker.getvalue()

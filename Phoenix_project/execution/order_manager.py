# execution/order_manager.py
"""
The core of the execution layer, responsible for translating a target portfolio
into concrete orders and managing their lifecycle.
"""
import logging
import math
from typing import List, Dict, Any
import backtrader as bt

from .interfaces import IBrokerAdapter, Order

class OrderManager:
    """
    Manages the rebalancing process by calculating, simulating, and placing orders.
    """
    def __init__(self,
                 broker_adapter: IBrokerAdapter,
                 impact_coefficient: float = 0.1,
                 max_volume_share: float = 0.25,
                 min_trade_notional: float = 100.0):
        self.logger = logging.getLogger("PhoenixProject.OrderManager")
        self.broker_adapter = broker_adapter
        self.impact_coefficient = impact_coefficient
        self.max_volume_share = max_volume_share
        self.min_trade_notional = min_trade_notional
        self.logger.info("OrderManager initialized.")

    def rebalance(self, strategy: bt.Strategy, target_portfolio: List[Dict[str, Any]]) -> None:
        self.logger.info("--- [Order Manager]: Rebalance protocol initiated ---")
        total_value = self.broker_adapter.get_portfolio_value()
        target_values = {item['ticker']: total_value * item['capital_allocation_pct'] for item in target_portfolio}
        
        current_positions = {d._name for d in strategy.datas if strategy.getposition(d).size != 0}
        all_tickers = sorted(list(current_positions.union(target_values.keys())))

        for ticker in all_tickers:
            data = strategy.getdatabyname(ticker)
            if not data:
                self.logger.warning(f"No data feed found for {ticker}; cannot execute.")
                continue

            current_value = strategy.getposition(data).size * data.close[0]
            target_value = target_values.get(ticker, 0.0)
            value_delta = target_value - current_value

            if abs(value_delta) < self.min_trade_notional:
                continue

            bar_volume = data.volume[0]
            if bar_volume == 0:
                self.logger.warning(f"No volume for {ticker}. Cannot trade.")
                continue

            ideal_size = value_delta / data.close[0]
            max_tradeable_size = bar_volume * self.max_volume_share
            final_size = math.copysign(min(abs(ideal_size), max_tradeable_size), ideal_size)
            
            if abs(ideal_size) > max_tradeable_size:
                self.logger.warning(f"Liquidity constraint for {ticker}: Ideal size {ideal_size:.0f} reduced to {final_size:.0f}")

            volume_share = abs(final_size) / bar_volume
            price_impact = self.impact_coefficient * (volume_share ** 0.5)

            side = 'BUY' if final_size > 0 else 'SELL'
            limit_price = data.close[0] * (1 + price_impact if side == 'BUY' else 1 - price_impact)

            order = Order(ticker=ticker, side=side, size=abs(final_size), limit_price=limit_price)
            self.broker_adapter.place_order(strategy, order)

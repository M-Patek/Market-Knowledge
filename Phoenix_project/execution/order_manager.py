# execution/order_manager.py
"""
The core of the execution layer, responsible for translating a target portfolio
into concrete orders and managing their lifecycle.
"""
import logging
import math
from typing import List, Dict, Any
import backtrader as bt

from .interfaces import Order

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

    def _calculate_ideal_size(self, current_value: float, target_value: float, current_price: float) -> float:
        """Calculates the ideal number of shares to trade to reach the target value."""
        value_delta = target_value - current_value
        if abs(value_delta) < self.min_trade_notional:
            return 0.0
        if current_price == 0:
            return 0.0
        return value_delta / current_price

    def _apply_liquidity_constraint(self, ideal_size: float, bar_volume: float, ticker: str) -> float:
        """Applies the max_volume_share constraint to the trade size."""
        if bar_volume == 0:
            self.logger.warning(f"No volume for {ticker}. Cannot trade.")
            return 0.0

        max_tradeable_size = bar_volume * self.max_volume_share
        if abs(ideal_size) > max_tradeable_size:
            self.logger.warning(f"Liquidity constraint for {ticker}: Ideal size {ideal_size:.0f} reduced to {math.copysign(max_tradeable_size, ideal_size):.0f}")
            return math.copysign(max_tradeable_size, ideal_size)
        return ideal_size

    def _estimate_limit_price(self, side: str, final_size: float, bar_volume: float, current_price: float) -> float:
        """Estimates the limit price based on a square root price impact model."""
        volume_share = abs(final_size) / bar_volume if bar_volume > 0 else 0
        price_impact = self.impact_coefficient * (volume_share ** 0.5)
        return current_price * (1 + price_impact if side == 'BUY' else 1 - price_impact)

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
            
            ideal_size = self._calculate_ideal_size(current_value, target_value, data.close[0])
            if ideal_size == 0:
                continue

            final_size = self._apply_liquidity_constraint(ideal_size, data.volume[0], ticker)
            if final_size == 0:
                continue

            side = 'BUY' if final_size > 0 else 'SELL'
            limit_price = self._estimate_limit_price(side, final_size, data.volume[0], data.close[0])
            
            order = Order(ticker=ticker, side=side, size=abs(final_size), limit_price=limit_price)
            self.broker_adapter.place_order(strategy, order)

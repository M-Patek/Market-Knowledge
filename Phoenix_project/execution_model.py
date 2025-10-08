# execution_model.py

import logging
import math
from typing import Protocol, List, Dict, Any

import backtrader as bt


class IExecutionModel(Protocol):
    """Defines the interface for an execution model."""

    def rebalance(self, strategy: bt.Strategy, target_portfolio: List[Dict[str, Any]]) -> None:
        """
        Receives a target portfolio allocation from the strategy and executes the
        necessary trades, considering market conditions like slippage and liquidity.
        """
        ...


class VolumeShareSlippageModel:
    """
    A realistic execution model that simulates slippage based on trade size
    relative to market volume and enforces partial fills based on liquidity constraints.
    """

    def __init__(self, impact_coefficient: float = 0.1, max_volume_share: float = 0.25, min_trade_notional: float = 100.0):
        """
        Args:
            impact_coefficient (float): A constant to control the magnitude of price impact.
            max_volume_share (float): The maximum percentage of a bar's volume a single
                                      order is allowed to consume.
            min_trade_notional (float): The minimum absolute dollar value for a trade to be considered.
        """
        self.logger = logging.getLogger("PhoenixProject.ExecutionModel")
        self.impact_coefficient = impact_coefficient
        self.max_volume_share = max_volume_share
        self.min_trade_notional = min_trade_notional
        self.logger.info(
            f"VolumeShareSlippageModel initialized. "
            f"Impact Coeff: {self.impact_coefficient}, Max Volume Share: {self.max_volume_share:.2%}, "
            f"Min Notional: ${self.min_trade_notional:,.2f}"
        )

    def rebalance(self, strategy: bt.Strategy, target_portfolio: List[Dict[str, Any]]) -> None:
        self.logger.info("--- [Execution Model]: Rebalance protocol initiated ---")
        total_value = strategy.broker.getvalue()
        target_values = {item['ticker']: total_value * item['capital_allocation_pct'] for item in target_portfolio}

        current_positions = {d._name for d in strategy.datas if strategy.getposition(d).size != 0}
        all_tickers = current_positions.union(target_values.keys())

        for ticker in sorted(list(all_tickers)): # Sort for deterministic order
            data = strategy.getdatabyname(ticker)
            if not data:
                self.logger.warning(f"No data feed found for {ticker}; cannot execute rebalance.")
                continue

            # Calculate the value difference to trade
            current_value = strategy.getposition(data).size * data.close[0]
            target_value = target_values.get(ticker, 0.0)
            value_delta = target_value - current_value

            # 1. Enforce Minimum Trade Size
            if abs(value_delta) < self.min_trade_notional:
                continue

            bar_volume = data.volume[0]
            if bar_volume == 0:
                self.logger.warning(f"No volume for {ticker} on {data.datetime.date(0)}. Cannot trade.")
                continue

            # 2. Calculate ideal order size and apply liquidity constraint (Partial Fill)
            ideal_size = value_delta / data.close[0]
            max_tradeable_size = bar_volume * self.max_volume_share
            
            final_size = ideal_size
            if abs(ideal_size) > max_tradeable_size:
                final_size = math.copysign(max_tradeable_size, ideal_size)
                self.logger.warning(f"Liquidity constraint for {ticker}: Ideal size {ideal_size:.0f} reduced to {final_size:.0f} (Max share: {self.max_volume_share:.2%})")

            # 3. Calculate price impact (Slippage)
            volume_share = abs(final_size) / bar_volume
            price_impact = self.impact_coefficient * (volume_share ** 0.5)

            if final_size > 0: # Buy Order
                limit_price = data.close[0] * (1 + price_impact)
                # 4. Simulate unfilled limit order
                if limit_price < data.low[0]:
                    self.logger.warning(f"BUY order for {ticker} likely unfilled. Limit Price ${limit_price:.2f} is below bar low ${data.low[0]:.2f}.")
                    continue
                strategy.buy(data=data, size=final_size, price=limit_price, exectype=bt.Order.Limit)
                self.logger.info(f"BUY order for {ticker}: Size={final_size:.2f}, Est. Limit Price=${limit_price:.2f} (Impact: +{price_impact:.4%})")
            elif final_size < 0: # Sell Order
                limit_price = data.close[0] * (1 - price_impact)
                # 4. Simulate unfilled limit order
                if limit_price > data.high[0]:
                    self.logger.warning(f"SELL order for {ticker} likely unfilled. Limit Price ${limit_price:.2f} is above bar high ${data.high[0]:.2f}.")
                    continue
                strategy.sell(data=data, size=final_size, price=limit_price, exectype=bt.Order.Limit)
                self.logger.info(f"SELL order for {ticker}: Size={final_size:.2f}, Est. Limit Price=${limit_price:.2f} (Impact: -{price_impact:.4%})")

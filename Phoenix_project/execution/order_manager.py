# execution/order_manager.py
import logging
import math
from typing import List, Dict, Any
import backtrader as bt
from execution.signal_protocol import StrategySignal # (L6) Import

from .interfaces import IBrokerAdapter, Order

class OrderManager:
    """
    (L6 Patched) Handles the creation, submission, and tracking of orders.
    It translates the desired portfolio state (StrategySignals) into concrete orders.
    """
    def __init__(self,
                 broker_adapter: IBrokerAdapter,
                 impact_coefficient: float = 0.1,
                 max_volume_share: float = 0.25,
                 min_trade_notional: float = 100.0,
                 average_spread_bps: float = 1.5,
                 fill_prob_aggressiveness: float = 10.0,
                 **kwargs): # Absorb extra config fields
        self.logger = logging.getLogger("PhoenixProject.OrderManager")
        self.broker_adapter = broker_adapter
        self.impact_coefficient = impact_coefficient
        self.max_volume_share = max_volume_share
        self.min_trade_notional = min_trade_notional
        self.average_spread_bps = average_spread_bps
        self.fill_prob_aggressiveness = fill_prob_aggressiveness
        self.execution_costs: List[float] = [] # [V2.0+] Store execution costs (slippage)
        self.logger.info("OrderManager initialized.")

    def _calculate_ideal_size(self, current_value: float, target_value: float, current_price: float) -> float:
        """Calculates the ideal number of shares to trade to reach the target value."""
        value_delta = target_value - current_value
        if abs(value_delta) < self.min_trade_notional:
            return 0.0
        if current_price == 0:
            return 0.0
        return value_delta / current_price

    def _constrain_size(self, ideal_size: float, bar_volume: float) -> float:
        """Constrains the order size based on the bar's volume."""
        max_size_from_volume = bar_volume * self.max_volume_share
        
        if abs(ideal_size) > max_size_from_volume:
            constrained_size = max_size_from_volume * (1 if ideal_size > 0 else -1)
            self.logger.warning(f"Order size constrained by volume. Ideal: {ideal_size:.2f}, Max Allowed: {max_size_from_volume:.2f}, Final: {constrained_size:.2f}")
            return constrained_size
        return ideal_size

    def _estimate_limit_price(self, side: str, final_size: float, bar_volume: float, current_price: float) -> float:
        """Estimates the limit price based on a square root price impact model plus spread cost."""
        # 1. Calculate price impact from order size
        volume_share = abs(final_size) / bar_volume if bar_volume > 0 else 0
        price_impact = self.impact_coefficient * (volume_share ** 0.5)
        # 2. Calculate cost of crossing half the bid-ask spread
        spread_impact = (self.average_spread_bps / 10000.0) / 2.0
        
        # 3. Add both costs to the current price
        return current_price * (1 + price_impact + spread_impact if side == 'BUY' else 1 - price_impact - spread_impact)

    def _calculate_fill_probability(self, side: str, limit_price: float, current_price: float) -> float:
        """Calculates the probability of a limit order being filled using a sigmoid function."""
        # How far is our price from the market, in basis points
        price_diff_bps = ((limit_price - current_price) / current_price) * 10000
        # For a buy, a higher limit price is more likely to fill (positive diff). For a sell, a lower price is (negative diff).
        advantage_bps = price_diff_bps if side == 'BUY' else -price_diff_bps
        return 1 / (1 + math.exp(-self.fill_prob_aggressiveness * (advantage_bps / 100))) # Normalize by 100

    def calculate_target_orders(self,
                                strategy_signals: List[StrategySignal], # (L6) Use StrategySignal
                                total_value: float,
                                current_positions: Dict[str, Dict[str, float]], # {ticker: {'size': s, 'value': v}}
                                market_data: Dict[str, Dict[str, float]] # {ticker: {'price': p, 'volume': v}}
                                ) -> List[Order]:
        """
        Calculates the list of orders required to adjust positions *in* the target portfolio.
        This method is framework-agnostic. It does NOT handle liquidation of assets
        not present in the target_portfolio; that is left to the caller.
        """
        orders_to_place: List[Order] = []

        # Adjust positions for assets in the target portfolio
        for signal in strategy_signals:
            ticker = signal.ticker
            target_pct = signal.target_weight # (L6) Use target_weight from signal
            if ticker not in market_data:
                self.logger.warning(f"No market data for target '{ticker}'. Skipping order calculation.")
                continue
            
            current_price = market_data[ticker]['price']
            current_volume = market_data[ticker]['volume']
            current_value = current_positions.get(ticker, {}).get('value', 0.0)
            
            target_value = total_value * target_pct
            
            ideal_size = self._calculate_ideal_size(current_value, target_value, current_price)
            final_size = self._constrain_size(ideal_size, current_volume)
            
            if final_size == 0:
                continue

            side = 'BUY' if final_size > 0 else 'SELL'
            limit_price = self._estimate_limit_price(side, final_size, current_volume, current_price)
            fill_prob = self._calculate_fill_probability(side, limit_price, current_price)
            
            orders_to_place.append(Order(
                ticker=ticker,
                side=side,
                size=abs(final_size),
                order_type='Limit',
                limit_price=limit_price,
                fill_probability=fill_prob
            ))
        
        return orders_to_place

    def rebalance(self, strategy: bt.Strategy, strategy_signals: List[StrategySignal]) -> None: # (L6) Use StrategySignal
        self.logger.info("--- [Order Manager]: Backtrader Rebalance protocol initiated ---")

        target_map = {signal.ticker: signal.target_weight for signal in strategy_signals} # (L6) Build map from signals
        total_value = strategy.broker.getvalue()

        # Step 1: (Unchanged) Liquidate or reduce positions no longer in the target portfolio
        for data in strategy.datas:
            ticker = data._name
            current_position = strategy.getposition(data)
            if current_position.size != 0 and ticker not in target_map:
                self.logger.info(f"'{ticker}' no longer a target. Closing position of {current_position.size} shares.")
                strategy.close(data)

        # Step 2: (Refactored) Adjust positions for assets in the target portfolio
        # Gather data for the pure calculation function
        current_positions = {}
        market_data = {}
        for data in strategy.datas:
            ticker = data._name
            if ticker not in strategy.getdatanames():
                continue
            pos = strategy.getposition(data)
            # Use data.close[0] for current price, as pos.price is the avg entry price
            current_price = data.close[0]
            current_positions[ticker] = {'size': pos.size, 'value': pos.size * current_price}
            market_data[ticker] = {'price': current_price, 'volume': data.volume[0]}

        # Call the framework-agnostic calculation method
        orders_to_place = self.calculate_target_orders(
            strategy_signals=strategy_signals, # (L6) Pass signals
            total_value=total_value,
            current_positions=current_positions,
            market_data=market_data
        )

        # Submit orders via the broker adapter
        for order in orders_to_place:
            self.logger.info(f"Submitting {order.side} order for '{order.ticker}': Size={order.size:.2f}, Est. Limit Price={order.limit_price:.2f}, Est. Fill Prob={order.fill_probability:.2%}")
            self.broker_adapter.place_order(strategy, order)

    def handle_order_notification(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Acknowledged by the broker, nothing to do.
            return

        if order.status in [order.Completed]:
            # [V2.0+] Cost Feedback Loop: Capture the cost of this execution
            slippage_bps = 0.0
            if order.exectype == order.ExecTypes.Limit and order.created.price:
                price_diff = order.executed.price - order.created.price
                # For BUY, a higher exec price is a cost. For SELL, a lower exec price is a cost.
                cost_sign = 1.0 if order.isbuy() else -1.0
                slippage_bps = (price_diff / order.created.price) * 10000 * cost_sign
                self.execution_costs.append(slippage_bps)

            self.logger.info(
                f"ORDER EXECUTED: {order.getstatusname()} - {order.info.ref} - {order.getordername()}: "
                f"{'BUY' if order.isbuy() else 'SELL'} {order.executed.size} {order.data._name} "
                f"@ {order.executed.price:.2f}, Value: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}, "
                f"Slippage: {slippage_bps:.2f} bps"
            )
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.logger.error(f"ORDER FAILED: {order.getstatusname()} - {order.info.ref} - {order.data._name}")

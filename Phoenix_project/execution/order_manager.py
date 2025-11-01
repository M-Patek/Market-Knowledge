import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

from .signal_protocol import StrategySignal
from ..core.pipeline_state import PipelineState

class Order:
    """
    Represents a single execution order to be sent to a broker.
    """
    def __init__(self, symbol: str, quantity: float, order_type: str = "MARKET", metadata: Dict = None):
        self.symbol = symbol
        self.quantity = quantity # Positive for buy, negative for sell
        self.order_type = order_type
        self.status = "NEW"
        self.metadata = metadata or {}

class OrderManager:
    """
    Responsible for translating a target portfolio (weights) into concrete
    execution orders (shares).
    
    It calculates the difference between the current portfolio and the target
    portfolio and generates the necessary trades.
    
    In a live-trading version, this module would also manage order execution,
    handle fills, and update the portfolio state. In this simulation,
    it just generates the orders.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.execution_config = config.get('execution_manager', {})
        self.base_capital = config.get('base_capital', 1_000_000)
        
        # Slippage and commission simulation
        self.simulate_costs = self.execution_config.get('simulate_costs', True)
        self.slippage_bps = self.execution_config.get('slippage_basis_points', 5)
        self.commission_per_share = self.execution_config.get('commission_per_share', 0.005)

    def generate_orders_from_signal(
        self, 
        signal: StrategySignal,
        state: PipelineState
    ) -> List[Order]:
        """
        Main entry point for the simulation.
        Generates orders required to move from the current state to the target signal.
        
        Args:
            signal (StrategySignal): The target portfolio state from the cognitive engine.
            state (PipelineState): The current state of the system (portfolio, market data).
            
        Returns:
            List[Order]: A list of orders to be "executed".
        """
        
        orders = []
        
        # Get total portfolio value (Equity = Cash + Positions Value)
        portfolio_value = self._calculate_portfolio_value(state)
        
        # Iterate over all assets defined in the signal (target portfolio)
        for symbol, target_weight in signal.target_weights.items():
            if symbol == 'CASH':
                continue # Cash is handled implicitly

            # 1. Calculate Target Position Value
            target_value = portfolio_value * target_weight
            
            # 2. Get Current Position Value
            current_shares = state.portfolio.get(symbol, 0.0)
            current_price = self._get_last_price(symbol, state)
            
            if current_price is None:
                # Can't trade if we don't have a price
                print(f"[OrderManager] Warning: No price data for {symbol}. Skipping order generation.")
                continue
                
            current_value = current_shares * current_price
            
            # 3. Calculate Delta
            value_to_trade = target_value - current_value
            
            # 4. Convert Delta Value to Delta Shares
            if abs(value_to_trade) > 0.01: # Avoid dust trades
                shares_to_trade = value_to_trade / current_price
                
                # Simple rounding (a real system would be more complex)
                quantity = round(shares_to_trade, 0) 
                
                if quantity != 0:
                    order = Order(
                        symbol=symbol,
                        quantity=quantity,
                        metadata={
                            "target_weight": target_weight,
                            "current_shares": current_shares,
                            "target_value": target_value,
                            "current_value": current_value,
                            "price": current_price
                        }
                    )
                    orders.append(order)
                    
        return orders

    def simulate_execution(
        self, 
        orders: List[Order], 
        state: PipelineState
    ) -> (List[Dict[str, Any]], Dict[str, float]):
        """
        Simulates the execution of orders, applying slippage and commissions.
        
        Returns:
            (List[Dict], Dict): A tuple of:
            1. A list of "fill" records (dicts).
            2. A dictionary of simulated costs (slippage, commissions).
        """
        if not self.simulate_costs:
            return [], {"slippage_cost": 0, "commission_cost": 0}

        fills = []
        total_slippage_cost = 0
        total_commission_cost = 0

        for order in orders:
            price = self._get_last_price(order.symbol, state)
            if price is None:
                continue
            
            # 1. Simulate Slippage
            # Slippage is worse for the trader (buy higher, sell lower)
            slippage_percent = (self.slippage_bps / 10000.0)
            if order.quantity > 0: # Buy Order
                fill_price = price * (1 + slippage_percent)
                slippage_cost = (fill_price - price) * order.quantity
            else: # Sell Order
                fill_price = price * (1 - slippage_percent)
                slippage_cost = (price - fill_price) * abs(order.quantity)
                
            total_slippage_cost += slippage_cost

            # 2. Simulate Commission
            commission = abs(order.quantity) * self.commission_per_share
            total_commission_cost += commission
            
            # 3. Create fill record
            fill_record = {
                "symbol": order.symbol,
                "quantity": order.quantity,
                "execution_price": fill_price,
                "ideal_price": price,
                "slippage_cost": slippage_cost,
                "commission_cost": commission,
                "timestamp": state.timestamp
            }
            fills.append(fill_record)
            
        costs = {
            "slippage_cost": total_slippage_cost,
            "commission_cost": total_commission_cost,
            "total_cost": total_slippage_cost + total_commission_cost
        }
        
        return fills, costs

    def _calculate_portfolio_value(self, state: PipelineState) -> float:
        """Calculates the total mark-to-market value of the portfolio."""
        value = state.portfolio.get('CASH', self.base_capital)
        
        for symbol, shares in state.portfolio.items():
            if symbol == 'CASH':
                continue
            price = self._get_last_price(symbol, state)
            if price is not None:
                value += shares * price
                
        return value

    def _get_last_price(self, symbol: str, state: PipelineState) -> Optional[float]:
        """Safely retrieves the last known price for a symbol."""
        if symbol in state.market_data:
            # Get the most recent TickerData object for this symbol
            # Assuming state.market_data[symbol] is a list of TickerData, sorted by time
            if state.market_data[symbol]:
                return state.market_data[symbol][-1].close
        
        return None # No price data available

"""
Execution Adapters

This module defines the interfaces (abstract base classes) for execution
and provides concrete implementations, such as:
- SimulatedBrokerAdapter: For backtesting, simulates order execution.
- LiveBrokerAdapter: (Example) For live trading, connects to a real broker API.
"""
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import uuid

# 修复：添加 pandas 导入
import pandas as pd

from .interfaces import Order, Fill, OrderStatus, ExecutionAdapter
from ..core.pipeline_state import PipelineState

logger = logging.getLogger(__name__)

class SimulatedBrokerAdapter(ExecutionAdapter):
    """
    A simulated broker for backtesting.
    
    It processes orders instantly (or with a delay) based on the
    next available market price, simulating fills, commissions, and slippage.
    """

    def __init__(self, 
                 pipeline_state: PipelineState,
                 commission_bps: float = 1.0, 
                 slippage_bps: float = 0.5):
        """
        Initializes the simulated broker.

        Args:
            pipeline_state: The central state, used to get current market prices.
            commission_bps: Commission fee in basis points (e.g., 1.0 = 0.01%).
            slippage_bps: Slippage in basis points (e.g., 0.5 = 0.005%).
        """
        self.pipeline_state = pipeline_state
        self.commission_rate = commission_bps / 10000.0
        self.slippage_rate = slippage_bps / 10000.0
        
        self.open_orders: Dict[str, Order] = {}
        logger.info(f"SimulatedBrokerAdapter initialized (Commission: {commission_bps}bps, Slippage: {slippage_bps}bps)")

    async def connect(self):
        """Simulated connection."""
        logger.info("SimulatedBrokerAdapter connecting...")
        await asyncio.sleep(0.01) # Simulate async connection
        logger.info("SimulatedBrokerAdapter connected.")

    async def disconnect(self):
        """Simulated disconnection."""
        logger.info("SimulatedBrokerAdapter disconnected.")

    async def submit_order(self, order: Order) -> OrderStatus:
        """
        Submits an order. In the simulation, this means checking if it
        can be filled immediately or queuing it.
        """
        if order.order_type == "MARKET":
            # Market orders are filled on the next tick (in the
            # TradeLifecycleManager), so we just store it.
            order.status = "ACCEPTED"
            order.order_id = str(uuid.uuid4())
            self.open_orders[order.order_id] = order
            
            logger.debug(f"Simulated order ACCEPTED: {order.order_id} ({order.symbol} {order.quantity})")
            return OrderStatus(
                order_id=order.order_id,
                status="ACCEPTED",
                timestamp=pd.Timestamp.now().to_pydatetime() # 修复：使用 pd.Timestamp
            )
        else:
            # Simulated broker only supports MARKET orders for simplicity
            logger.warning(f"Order type {order.order_type} not supported by sim broker.")
            return OrderStatus(
                order_id=order.order_id,
                status="REJECTED",
                timestamp=pd.Timestamp.now().to_pydatetime() # 修复：使用 pd.Timestamp
            )

    async def cancel_order(self, order_id: str) -> OrderStatus:
        """Cancels an open order."""
        if order_id in self.open_orders:
            order = self.open_orders.pop(order_id)
            order.status = "CANCELLED"
            logger.debug(f"Simulated order CANCELLED: {order_id}")
            return OrderStatus(
                order_id=order_id,
                status="CANCELLED",
                timestamp=pd.Timestamp.now().to_pydatetime() # 修复：使用 pd.Timestamp
            )
        else:
            logger.warning(f"Attempted to cancel non-existent order_id: {order_id}")
            return OrderStatus(
                order_id=order_id,
                status="REJECTED",
                message="Order ID not found.",
                timestamp=pd.Timestamp.now().to_pydatetime() # 修复：使用 pd.Timestamp
            )

    async def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        """Gets the status of a specific order."""
        if order_id in self.open_orders:
            order = self.open_orders[order_id]
            return OrderStatus(
                order_id=order_id,
                status=order.status,
                timestamp=order.timestamp # Assumes Order has a timestamp
            )
        # In a real sim, we'd also check filled/cancelled orders
        logger.warning(f"Order ID {order_id} not found in open orders.")
        return None

    def get_fill_price_and_cost(self, order: Order) -> Tuple[float, float, float]:
        """
        Simulates the fill price and cost for a market order.
        This is called by the TradeLifecycleManager.
        
        Returns:
            (fill_price, commission, total_cost)
        """
        current_price = self.pipeline_state.get_market_price(order.symbol)
        
        if current_price is None:
            raise ValueError(f"No market price available for {order.symbol} to fill order.")
            
        # 1. Apply Slippage
        if order.quantity > 0: # Buying
            fill_price = current_price * (1 + self.slippage_rate)
        else: # Selling
            fill_price = current_price * (1 - self.slippage_rate)
            
        # 2. Calculate Cost and Commission
        gross_cost = fill_price * order.quantity
        commission = abs(gross_cost) * self.commission_rate
        
        # Total cost (if buying, cost is positive; if selling, cost is negative)
        # Commission is always a positive cost (reduces cash)
        total_cost = gross_cost
        
        return fill_price, commission, total_cost

    def process_fill(self, order: Order, fill_price: float, commission: float):
        """
        Finalizes an order, marks it as FILLED, and removes it from open_orders.
        """
        if order.order_id in self.open_orders:
            self.open_orders.pop(order.order_id)
        
        order.status = "FILLED"
        
        fill_event = Fill(
            order_id=order.order_id,
            timestamp=self.pipeline_state.get_current_time(), # Fill at current time
            symbol=order.symbol,
            quantity=order.quantity,
            fill_price=fill_price,
            commission=commission
        )
        
        logger.debug(f"Simulated order FILLED: {order.order_id} ({order.symbol} @ {fill_price:.2f})")
        return fill_event

# --- Example Live Adapter (Stub) ---

class LiveBrokerAdapter(ExecutionAdapter):
    """
    (Stub) Example of a live broker adapter.
    This would wrap a real API (e.g., Alpaca, IBKR).
    """

    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        # self.api_client = ThirdPartyBrokerAPI(api_key, api_secret)
        logger.info("LiveBrokerAdapter initialized (STUB).")

    async def connect(self):
        # await self.api_client.connect()
        logger.info("LiveBrokerAdapter connected (STUB).")

    async def disconnect(self):
        # await self.api_client.disconnect()
        logger.info("LiveBrokerAdapter disconnected (STUB).")

    async def submit_order(self, order: Order) -> OrderStatus:
        logger.info(f"Submitting LIVE order: {order.symbol} {order.quantity} (STUB)")
        # live_order = await self.api_client.submit(...)
        # return OrderStatus(order_id=live_order.id, status=live_order.status, ...)
        raise NotImplementedError("LiveBrokerAdapter is a stub and not implemented.")

    async def cancel_order(self, order_id: str) -> OrderStatus:
        logger.info(f"Cancelling LIVE order: {order_id} (STUB)")
        # ...
        raise NotImplementedError("LiveBrokerAdapter is a stub and not implemented.")

    async def get_order_status(self, order_id: str) -> Optional[OrderStatus]:
        logger.info(f"Getting LIVE order status: {order_id} (STUB)")
        # ...
        raise NotImplementedError("LiveBrokerAdapter is a stub and not implemented.")

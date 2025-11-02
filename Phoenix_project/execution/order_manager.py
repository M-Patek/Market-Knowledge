from typing import Dict, Any, List
import asyncio
from core.schemas.data_schema import Order, Execution
from core.pipeline_state import PipelineState
from events.event_distributor import EventDistributor
from execution.adapters import BrokerAdapter
from monitor.logging import get_logger

logger = get_logger(__name__)

class OrderManager:
    """
    Manages the full lifecycle of orders:
    - Receives pending orders from PortfolioConstructor.
    - Submits them via the BrokerAdapter.
    - Listens for execution fills.
    - Updates the PipelineState (portfolio) based on fills.
    """

    def __init__(
        self,
        pipeline_state: PipelineState,
        event_distributor: EventDistributor,
        broker_adapter: BrokerAdapter,
        config: Dict[str, Any]
    ):
        self.pipeline_state = pipeline_state
        self.event_distributor = event_distributor
        self.broker_adapter = broker_adapter
        self.config = config
        
        # In-memory tracking of orders
        self._open_orders: Dict[str, Order] = {} # {order_id: Order}
        
        logger.info(f"OrderManager initialized with adapter: {broker_adapter.__class__.__name__}")

    async def initialize(self):
        """Initialize the broker adapter and subscribe to events."""
        try:
            await self.broker_adapter.initialize()
            # Subscribe to execution events published by the adapter
            await self.event_distributor.subscribe("order_execution", self.on_execution)
            logger.info("OrderManager initialized and subscribed to executions.")
        except Exception as e:
            logger.error(f"Failed to initialize OrderManager: {e}", exc_info=True)
            raise

    async def submit_order(self, order: Order):
        """Submits a new order to the broker."""
        if order.id in self._open_orders:
            logger.warning(f"Order {order.id} is already being tracked.")
            return
            
        try:
            logger.info(f"Submitting order {order.id} to broker...")
            submitted_order = await self.broker_adapter.submit_order(order)
            
            if submitted_order.status == "REJECTED":
                logger.error(f"Order {order.id} was REJECTED by broker: {submitted_order.reject_reason}")
                await self.event_distributor.publish(
                    "order_rejected", order=submitted_order
                )
            elif submitted_order.status == "FILLED":
                logger.info(f"Order {order.id} was filled immediately (simulated).")
                # The adapter should have already published this event
                pass
            else: # e.g., "PENDING", "ACCEPTED"
                logger.info(f"Order {order.id} is open with status: {submitted_order.status}")
                self._open_orders[submitted_order.id] = submitted_order
                
        except Exception as e:
            logger.error(f"Error submitting order {order.id}: {e}", exc_info=True)
            order.status = "REJECTED"
            order.reject_reason = f"Submission error: {e}"
            await self.event_distributor.publish("order_rejected", order=order)

    async def on_execution(self, pipeline_state: PipelineState, execution: Execution):
        """
        Callback for 'order_execution' events.
        This is the most critical part: updating the portfolio state.
        """
        logger.info(f"Received execution: {execution.quantity} {execution.symbol} @ {execution.fill_price}")
        
        try:
            # 1. Update the order status
            if execution.order_id in self._open_orders:
                order = self._open_orders[execution.order_id]
                # TODO: Handle partial fills
                order.filled_quantity += execution.quantity
                order.status = "FILLED" # Simplified
                del self._open_orders[execution.order_id]
            else:
                logger.warning(f"Received execution for untracked/already-filled order: {execution.order_id}")
            
            # 2. Update trade history
            await self.pipeline_state.update_state({"trade_history": [execution]})
            
            # 3. Update portfolio (the hard part)
            portfolio = await self.pipeline_state.get_portfolio()
            positions = portfolio.get("positions", {})
            cash = portfolio.get("cash", 0.0)
            
            symbol = execution.symbol
            quantity = execution.quantity
            fill_price = execution.fill_price
            commission = execution.commission
            
            # --- Update Cash ---
            trade_cost = (quantity * fill_price) # Negative for buys, positive for sells
            cash -= trade_cost
            cash -= commission
            
            # --- Update Position ---
            position = positions.get(symbol, {"size": 0, "avg_price": 0, "market_value": 0})
            
            current_size = position["size"]
            current_avg_price = position["avg_price"]
            
            new_size = current_size + quantity
            
            if new_size == 0:
                # Position closed
                new_avg_price = 0
            elif (current_size * quantity) >= 0: # Adding to position (or opening)
                # Weighted average price
                new_avg_price = ((current_avg_price * current_size) + (fill_price * quantity)) / new_size
            else:
                # Reducing position (realizing PnL)
                # Average price does not change
                new_avg_price = current_avg_price
                
            position["size"] = new_size
            position["avg_price"] = new_avg_price
            
            positions[symbol] = position
            
            # --- Recalculate Portfolio Value ---
            # This requires getting *all* current market prices
            # For simplicity, we'll just update cash and positions
            # A separate "PortfolioManager" service should update market values.
            
            new_portfolio_state = {
                "cash": cash,
                "positions": positions
                # "total_value" should be updated by another process
            }
            
            await self.pipeline_state.update_portfolio(new_portfolio_state)
            
            logger.info(f"Portfolio updated for {symbol}: New Size {new_size}, New Cash {cash:.2f}")

        except Exception as e:
            logger.error(f"Failed to process execution {execution.execution_id}: {e}", exc_info=True)
            # This is a critical error!
            # await self.event_distributor.publish("CRITICAL_ERROR", ...)

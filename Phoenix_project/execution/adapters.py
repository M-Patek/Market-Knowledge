from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import asyncio
import uuid
from datetime import datetime
from core.schemas.data_schema import Order, Execution
from core.pipeline_state import PipelineState
from monitor.logging import get_logger

logger = get_logger(__name__)

class BrokerAdapter(ABC):
    """Abstract base class for all broker execution adapters."""
    
    @abstractmethod
    async def initialize(self):
        """Initialize connection to the broker."""
        pass

    @abstractmethod
    async def submit_order(self, order: Order) -> Order:
        """Submit an order to the broker."""
        pass
        
    @abstractmethod
    async def cancel_order(self, order_id: str) -> Order:
        """Cancel an open order."""
        pass
        
    @abstractmethod
    async def get_order_status(self, order_id: str) -> Order:
        """Check the status of a specific order."""
        pass
        
    @abstractmethod
    async def get_account_summary(self) -> Dict[str, Any]:
        """Get account info (buying power, cash, etc.)."""
        pass
        
    @abstractname
    async def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions."""
        pass


class SimulatedBrokerAdapter(BrokerAdapter):
    """
    A simulated broker that mimics order execution for backtesting
    or paper trading.
    """
    
    def __init__(self, pipeline_state: PipelineState, config: Dict[str, Any]):
        self.pipeline_state = pipeline_state
        self.config = config.get("simulated_broker", {})
        self.slippage_pct = self.config.get("slippage_pct", 0.001) # 0.1% slippage
        self.commission_per_share = self.config.get("commission_per_share", 0.005)
        
        # Internal state
        self._open_orders: Dict[str, Order] = {}
        logger.info("SimulatedBrokerAdapter initialized.")

    async def initialize(self):
        logger.info("SimulatedBrokerAdapter: No initialization required.")
        await asyncio.sleep(0) # Be async
    
    async def submit_order(self, order: Order) -> Order:
        """
        Simulates the execution of an order.
        In this simple simulation, the order is filled *immediately*
        at the current market price + slippage.
        """
        logger.info(f"SimBroker: Received order {order.id} for {order.quantity} {order.symbol}")
        
        market_data = await self.pipeline_state.get_latest_market_data(order.symbol)
        if not market_data:
            logger.error(f"SimBroker: Cannot fill order {order.id}, no market data.")
            order.status = "REJECTED"
            order.reject_reason = "No market data"
            return order
            
        current_price = market_data.close
        
        # 1. Calculate Slippage
        if order.quantity > 0: # Buy
            fill_price = current_price * (1 + self.slippage_pct)
        else: # Sell
            fill_price = current_price * (1 - self.slippage_pct)
            
        # 2. Calculate Commission
        commission = abs(order.quantity) * self.commission_per_share
        
        # 3. Create Execution
        execution = Execution(
            execution_id=f"exec_{uuid.uuid4()}",
            order_id=order.id,
            timestamp=datetime.utcnow(),
            symbol=order.symbol,
            quantity=order.quantity,
            fill_price=fill_price,
            commission=commission
        )
        
        order.status = "FILLED"
        order.filled_quantity = execution.quantity
        order.avg_fill_price = execution.fill_price
        
        logger.info(f"SimBroker: Order {order.id} FILLED at {fill_price:.2f}")
        
        # 4. Publish execution event so portfolio can be updated
        # This is crucial.
        await self.pipeline_state.event_distributor.publish(
            "order_execution", execution=execution
        )
        
        return order
        
    async def cancel_order(self, order_id: str) -> Order:
        """
        In this simple simulation, orders are filled instantly,
        so they can't be cancelled.
        """
        if order_id in self._open_orders:
            order = self._open_orders.pop(order_id)
            order.status = "CANCELLED"
            logger.info(f"SimBroker: Cancelled order {order_id}")
            return order
        
        logger.warning(f"SimBroker: Could not cancel order {order_id}, not found or already filled.")
        # Need to fetch the filled order status
        return await self.get_order_status(order_id) # Placeholder

    async def get_order_status(self, order_id: str) -> Order:
        """
        Since orders fill instantly, we check the trade history
        (which is updated by the OrderManager).
        This is a bit complex and shows the coupling.
        """
        logger.warning("SimBroker.get_order_status is not fully implemented.")
        # A real sim would look up its internal state.
        return self._open_orders.get(order_id) # Wrong, as it's removed on fill

    async def get_account_summary(self) -> Dict[str, Any]:
        """Gets account summary from the PipelineState."""
        portfolio = await self.pipeline_state.get_portfolio()
        return {
            "cash": portfolio.get("cash"),
            "buying_power": portfolio.get("cash"), # Simplified
            "total_value": portfolio.get("total_value")
        }

    async def get_open_positions(self) -> List[Dict[str, Any]]:
        """Gets positions from the PipelineState."""
        portfolio = await self.pipeline_state.get_portfolio()
        positions = portfolio.get("positions", {})
        # Format as a list
        return [
            {"symbol": symbol, **pos_data}
            for symbol, pos_data in positions.items()
            if pos_data.get("size", 0) != 0
        ]


class LiveBrokerAdapter(BrokerAdapter):
    """
    Adapter for a live brokerage (e.g., Alpaca, IBKR).
    This is a placeholder.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_client = None # e.g., alpaca_trade_api.REST
        logger.warning("LiveBrokerAdapter is a placeholder and not functional.")

    async def initialize(self):
        logger.info("LiveBrokerAdapter: Initializing connection...")
        # try:
        #     self.api_client = alpaca_trade_api.REST(
        #         key_id=self.config["api_key"],
        #         secret_key=self.config["api_secret"],
        #         base_url=self.config["base_url"]
        #     )
        #     account = self.api_client.get_account()
        #     logger.info(f"LiveBrokerAdapter: Connection successful. Account: {account.id}")
        # except Exception as e:
        #     logger.error(f"Failed to connect to live broker: {e}")
        #     raise
        await asyncio.sleep(0) # Placeholder

    async def submit_order(self, order: Order) -> Order:
        logger.warning("LiveBrokerAdapter.submit_order is not implemented.")
        order.status = "REJECTED"
        order.reject_reason = "Live adapter not implemented"
        return order
        
    async def cancel_order(self, order_id: str) -> Order:
        logger.warning("LiveBrokerAdapter.cancel_order is not implemented.")
        raise NotImplementedError
        
    async def get_order_status(self, order_id: str) -> Order:
        logger.warning("LiveBrokerAdapter.get_order_status is not implemented.")
        raise NotImplementedError
        
    async def get_account_summary(self) -> Dict[str, Any]:
        logger.warning("LiveBrokerAdapter.get_account_summary is not implemented.")
        raise NotImplementedError
        
    async def get_open_positions(self) -> List[Dict[str, Any]]:
        logger.warning("LiveBrokerAdapter.get_open_positions is not implemented.")
        raise NotImplementedError

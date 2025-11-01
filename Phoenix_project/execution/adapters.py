from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import asyncio

# 修复：将根目录导入 'context_bus' 改为相对导入
from ..context_bus import ContextBus
from .interfaces import Order, Fill, MarketData
from ..monitor.logging import get_logger

logger = get_logger(__name__)

class BaseExecutionAdapter(ABC):
    """
    Abstract Base Class for all execution adapters (e.g., paper trading, real brokers).
    Connects to the ContextBus to listen for orders and publish fills.
    """
    
    def __init__(self, 
                 context_bus: ContextBus,
                 config: Dict[str, Any]):
        self.context_bus = context_bus
        self.config = config
        self.adapter_id = config.get('adapter_id', 'base_adapter')
        self._running = False
        self._order_listener_task = None
        
        logger.info(f"Initialized execution adapter: {self.adapter_id}")

    @abstractmethod
    async def connect(self):
        """
        Establish connection to the broker/exchange endpoint.
        """
        pass
        
    @abstractmethod
    async def disconnect(self):
        """
        Disconnect from the broker/exchange.
        """
        pass
        
    @abstractmethod
    async def submit_order(self, order: Order) -> str:
        """
        Submit an order to the execution venue.
        
        Returns:
            str: A unique broker-side order ID.
        """
        pass
        
    @abstractmethod
    async def cancel_order(self, broker_order_id: str) -> bool:
        """
        Cancel an existing order.
        """
        pass
        
    @abstractmethod
    async def get_order_status(self, broker_order_id: str) -> Dict[str, Any]:
        """
        Get the current status of an order.
        """
        pass
        
    @abstractmethod
    async def get_account_summary(self) -> Dict[str, Any]:
        """
        Get account balance, positions, etc.
        """
        pass

    async def _on_fill(self, fill: Fill):
        """
        Internal handler to publish a fill event to the ContextBus.
        """
        logger.info(f"Publishing fill: {fill.symbol} {fill.fill_amount} @ {fill.fill_price}")
        await self.context_bus.publish('fills', fill.model_dump())
        
    async def _on_market_data(self, market_data: MarketData):
        """
        Internal handler to publish market data to the ContextBus.
        """
        await self.context_bus.publish('market_data', market_data.model_dump())

    async def _order_listener(self):
        """
        Background task that listens for new orders on the ContextBus.
        """
        logger.info(f"Adapter {self.adapter_id} starting order listener...")
        try:
            # Subscribe to the 'orders' channel
            async for order_data in self.context_bus.subscribe('orders'):
                if not self._running:
                    break
                
                try:
                    order = Order(**order_data)
                    logger.debug(f"Adapter received order: {order.order_id}")
                    
                    # (Optional) Filter orders meant for this adapter
                    # if order.adapter_id != self.adapter_id:
                    #     continue
                        
                    broker_id = await self.submit_order(order)
                    logger.info(f"Order {order.order_id} submitted. Broker ID: {broker_id}")
                    # (We'd need to map broker_id back to order.order_id)
                    
                except Exception as e:
                    logger.error(f"Failed to process order: {order_data}, Error: {e}")

        except asyncio.CancelledError:
            logger.info(f"Order listener for {self.adapter_id} cancelled.")
        except Exception as e:
            logger.error(f"Error in order listener: {e}", exc_info=True)
            
    async def start(self):
        """
        Connects the adapter and starts its listeners.
        """
        if self._running:
            logger.warning(f"Adapter {self.adapter_id} is already running.")
            return
            
        try:
            await self.connect()
            self._running = True
            self._order_listener_task = asyncio.create_task(self._order_listener())
            logger.info(f"Adapter {self.adapter_id} started successfully.")
        except Exception as e:
            logger.error(f"Failed to start adapter {self.adapter_id}: {e}")
            self._running = False

    async def stop(self):
        """
        Stops the listeners and disconnects the adapter.
        """
        if not self._running:
            logger.warning(f"Adapter {self.adapter_id} is not running.")
            return
            
        logger.info(f"Stopping adapter {self.adapter_id}...")
        self._running = False
        
        if self._order_listener_task:
            self._order_listener_task.cancel()
            try:
                await self._order_listener_task
            except asyncio.CancelledError:
                pass
                
        await self.disconnect()
        logger.info(f"Adapter {self.adapter_id} stopped.")


class PaperTradingAdapter(BaseExecutionAdapter):
    """
    A simulation adapter that mimics a real broker.
    It simulates fills based on incoming market data.
    """
    
    def __init__(self, context_bus: ContextBus, config: Dict[str, Any]):
        super().__init__(context_bus, config)
        self.adapter_id = config.get('adapter_id', 'paper_trader')
        self.open_orders: Dict[str, Order] = {} # Keyed by broker_order_id
        self.market_data_listener_task = None
        self.last_market_prices: Dict[str, float] = {}

    async def connect(self):
        # No real connection needed for paper trading
        logger.info("PaperTradingAdapter 'connected'.")
        await asyncio.sleep(0)

    async def disconnect(self):
        # No real disconnection needed
        logger.info("PaperTradingAdapter 'disconnected'.")
        await asyncio.sleep(0)

    async def submit_order(self, order: Order) -> str:
        broker_order_id = f"PAPER-{order.order_id}"
        
        # If it's a market order, we need to wait for the next tick
        if order.order_type == "MARKET":
            self.open_orders[broker_order_id] = order
            
        # (Add logic for LIMIT orders)
        
        return broker_order_id

    async def cancel_order(self, broker_order_id: str) -> bool:
        if broker_order_id in self.open_orders:
            del self.open_orders[broker_order_id]
            logger.info(f"Paper order cancelled: {broker_order_id}")
            return True
        return False

    async def get_order_status(self, broker_order_id: str) -> Dict[str, Any]:
        if broker_order_id in self.open_orders:
            return {"status": "OPEN", "filled": 0}
        return {"status": "UNKNOWN", "filled": 0} # Assume filled or cancelled if not open

    async def get_account_summary(self) -> Dict[str, Any]:
        # (This should be managed by a separate PortfolioManager,
        # but we can provide a dummy summary)
        return {"balance": 1_000_000, "positions": {}}

    async def _market_data_listener(self):
        """
        Listens to market data to simulate fills for open orders.
        """
        logger.info(f"Paper adapter starting market data listener...")
        try:
            async for data in self.context_bus.subscribe('market_data'):
                if not self._running:
                    break
                
                market_data = MarketData(**data)
                self.last_market_prices[market_data.symbol] = market_data.close # Use close as fill price
                
                # Check if this tick can fill any open orders
                filled_orders = []
                for broker_id, order in self.open_orders.items():
                    if order.symbol == market_data.symbol:
                        # Simulate fill
                        fill = Fill(
                            order_id=order.order_id,
                            symbol=order.symbol,
                            fill_amount=order.amount,
                            fill_price=market_data.close, # No slippage
                            timestamp=market_data.timestamp
                        )
                        await self._on_fill(fill)
                        filled_orders.append(broker_id)
                
                # Remove filled orders
                for broker_id in filled_orders:
                    del self.open_orders[broker_id]
        
        except asyncio.CancelledError:
            logger.info(f"Market data listener for {self.adapter_id} cancelled.")
        except Exception as e:
            logger.error(f"Error in market data listener: {e}", exc_info=True)

    async def start(self):
        # Start the order listener (from base class)
        await super().start() 
        # Start the market data listener (specific to this class)
        self.market_data_listener_task = asyncio.create_task(self._market_data_listener())

    async def stop(self):
        # Stop the order listener
        await super().stop()
        # Stop the market data listener
        if self.market_data_listener_task:
            self.market_data_listener_task.cancel()
            try:
                await self.market_data_listener_task
            except asyncio.CancelledError:
                pass


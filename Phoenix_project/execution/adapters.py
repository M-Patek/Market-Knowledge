from abc import ABC, abstractmethod
from typing import List, Dict, Any

from .interfaces import BrokerInterface, DataInterface
from ..core.schemas.data_schema import TickerData, MarketEvent
from .order_manager import Order

# --- Example Broker Adapter ---

class SimulatedBrokerAdapter(BrokerInterface):
    """
    A simulated broker that mimics a real brokerage API (like Alpaca or IBKR).
    In a backtest, this is instantiated by the OrderManager.
    In live trading, this would wrap a real API client.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_key = config.get('api_key')
        self.api_secret = config.get('api_secret')
        self.base_url = config.get('base_url')
        
    async def connect(self):
        print(f"[SimBroker] Connecting to {self.base_url}...")
        # In a real adapter, this would initialize the API client
        print("[SimBroker] Connection successful (simulation).")
        
    async def submit_order(self, order: Order) -> Dict[str, Any]:
        """Simulates submitting an order."""
        print(f"[SimBroker] Submitting order: {order.quantity} {order.symbol} @ {order.order_type}")
        # In a real adapter, this returns the broker's order ID
        return {
            "id": f"sim_{order.symbol}_{pd.Timestamp.now().to_nsec()}",
            "status": "accepted",
            "symbol": order.symbol,
            "qty": order.quantity
        }
        
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Simulates checking an order status."""
        # In simulation, we assume immediate fill
        return {
            "id": order_id,
            "status": "filled",
            "filled_qty": "100", # Example
            "filled_avg_price": "150.25" # Example
        }

    async def get_account_info(self) -> Dict[str, Any]:
        """Simulates fetching account data."""
        return {
            "account_number": "SIM_123",
            "cash": "100000.00",
            "equity": "100000.00",
            "status": "ACTIVE"
        }
        
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Simulates fetching current positions."""
        # This would be dynamic in a real broker
        return [
            # {"symbol": "AAPL", "qty": "50", "avg_entry_price": "145.00"}
        ]


# --- Example Data Adapter ---

class YFinanceDataStream(DataInterface):
    """
    An adapter that implements the DataInterface using yfinance.
    (Note: yfinance is not a real-time stream, this is just for
    demonstrating the adapter pattern).
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    async def connect(self, on_message_callback):
        """
        Simulates a connection.
        In a real stream, this would start the websocket listener.
        """
        print("[YFinanceDataStream] 'Connecting' (simulation)...")
        self.on_message = on_message_callback
        print("[YFinanceDataStream] Ready.")
        
    async def subscribe_to_symbols(self, symbols: List[str]):
        """
        Simulates subscribing to symbols.
        Since yfinance is not a stream, this is a no-op.
        """
        print(f"[YFinanceDataStream] 'Subscribed' to {symbols} (simulation).")
        
    async def _simulate_message(self):
        """Helper to simulate a message arriving."""
        # This would be called by the underlying websocket client
        raw_data = {
            "type": "price",
            "symbol": "AAPL",
            "price": 150.00,
            "volume": 1000,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        await self.on_message(raw_data)

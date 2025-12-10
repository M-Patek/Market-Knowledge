"""
Phoenix_project/execution/order_manager.py
[Phase 3 Task 3] Fix OrderManager Zombie State.
Implement on_order_update to clean up active_orders on terminal states.
Unblock 'One-Shot' deadlock.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from Phoenix_project.execution.interfaces import IBrokerAdapter
from Phoenix_project.execution.trade_lifecycle_manager import TradeLifecycleManager
from Phoenix_project.core.schemas.data_schema import Order, OrderStatus, OrderType, OrderSide
from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.data_manager import DataManager

logger = logging.getLogger(__name__)

class OrderManager:
    """
    订单管理器 (Order Manager)
    负责将目标投资组合转换为具体的订单，并管理订单生命周期。
    
    [Fix] 实时监听订单状态，自动清理已完成/取消的订单，防止死锁。
    """

    def __init__(
        self, 
        broker: IBrokerAdapter, 
        trade_lifecycle_manager: TradeLifecycleManager,
        data_manager: DataManager,
        bus: Any = None # [Task 1.2] Inject ContextBus
    ):
        self.broker = broker
        self.tlm = trade_lifecycle_manager
        self.data_manager = data_manager
        self.bus = bus
        
        # 内部跟踪: symbol -> order_id (防止对同一标的重复下单)
        # 也可以是 order_id -> Order object
        self.active_orders: Dict[str, str] = {} 
        self._lock = asyncio.Lock()
        
        # [Task 3.3] Subscribe to Broker Updates
        # Ensure we listen to Status Updates (Canceled, Rejected, etc.) AND Fills
        if self.broker:
            self.broker.subscribe_order_status(self.on_order_update)
        
        # Connect explicitly if needed (Broker might handle lazy connect)
        # asyncio.create_task(self.broker.connect()) # Moved to Orchestrator or explicit start
        
        logger.info("OrderManager initialized and subscribed to broker updates.")

    async def on_order_update(self, update: Any):
        """
        [Task 3.3] Callback for Order Status Updates.
        Cleans up active_orders when orders reach terminal state.
        """
        try:
            # Normalize update data (Alpaca stream returns objects or dicts)
            # update.event: new, fill, partial_fill, canceled, expired, replaced, rejected, pending_cancel, etc.
            # update.order: { id: ..., symbol: ... }
            
            # Helper to extract data safely
            event = getattr(update, 'event', None)
            order_data = getattr(update, 'order', None)
            
            if not event or not order_data:
                # Fallback for dict
                if isinstance(update, dict):
                    event = update.get('event')
                    order_data = update.get('order')
            
            if not event or not order_data:
                return

            # Extract IDs
            # Alpaca object uses attributes, dict uses keys
            order_id = getattr(order_data, 'id', None) or order_data.get('id')
            symbol = getattr(order_data, 'symbol', None) or order_data.get('symbol')
            
            if not order_id or not symbol:
                return

            logger.info(f"Order Update: {symbol} [{order_id}] -> {event}")

            # Define Terminal States
            TERMINAL_STATES = {
                'filled', 'canceled', 'expired', 'rejected', 
                'stopped', 'done_for_day', 'replaced' # Replaced creates new order, old one is dead
            }

            if event in TERMINAL_STATES:
                async with self._lock:
                    # Check if this order is tracked as active
                    current_active_id = self.active_orders.get(symbol)
                    
                    if current_active_id == str(order_id):
                        del self.active_orders[symbol]
                        logger.info(f"Order {order_id} for {symbol} finished ({event}). Removed from active tracking.")
                    
                    # Handle 'replaced' logic specifically if needed?
                    # For now, we assume Orchestrator will issue new orders if needed.

            # [Task 1.2] Pass 'fill' events to TradeLifecycleManager and Publish Event
            if event == 'fill' or event == 'partial_fill':
                # Extract fill details
                filled_qty = getattr(order_data, 'filled_qty', 0) or order_data.get('filled_qty', 0)
                filled_avg_price = getattr(order_data, 'filled_avg_price', 0) or order_data.get('filled_avg_price', 0)
                
                payload = {
                    "symbol": symbol,
                    "order_id": order_id,
                    "status": event,
                    "filled_qty": float(filled_qty),
                    "filled_avg_price": float(filled_avg_price),
                    "timestamp": datetime.now().isoformat()
                }

                # 1. Notify Ledger (TLM)
                if self.tlm:
                    await self.tlm.on_fill(payload) # Assuming payload matches expected Fill format or TLM adapts
                
                # 2. Publish to Bus
                if self.bus:
                    await self.bus.publish("ORDER_FILLED", payload)

            # Publish Failure Events
            elif event in {'canceled', 'rejected', 'expired', 'failed'}:
                if self.bus:
                    await self.bus.publish("ORDER_FAILED", {
                        "symbol": symbol,
                        "order_id": order_id,
                        "status": event,
                        "timestamp": datetime.now().isoformat()
                    })

        except Exception as e:
            logger.error(f"Error processing order update: {e}", exc_info=True)

    async def reconcile_portfolio(self, state: PipelineState):
        """
        Reconciles the current portfolio with the target portfolio.
        Generates and places orders.
        """
        target_portfolio = state.target_portfolio
        if not target_portfolio:
            return

        logger.info("Reconciling portfolio...")
        
        # 1. Get Current Positions (from Broker or Ledger)
        # Using Broker for execution accuracy
        if not self.broker:
             logger.warning("No broker adapter configured. Skipping reconciliation.")
             return

        account_info = self.broker.get_account_info()
        if account_info.get("status") == "error":
            logger.error(f"Cannot reconcile: {account_info.get('message')}")
            return
            
        current_positions = {
            p['symbol']: float(p['qty']) 
            for p in account_info.get('positions', [])
        }
        
        # 2. Calculate Diffs
        target_positions = target_portfolio.get('positions', {})
        
        # Symbols to trade: Union of current and target
        all_symbols = set(current_positions.keys()) | set(target_positions.keys())
        
        async with self._lock:
            for symbol in all_symbols:
                # Skip if there is an active order for this symbol
                if symbol in self.active_orders:
                    logger.info(f"Skipping {symbol}: Active order {self.active_orders[symbol]} pending.")
                    continue
                
                current_qty = current_positions.get(symbol, 0.0)
                # Target format check: target_portfolio might be complex object or dict
                # Assuming dict: {symbol: {'quantity': x}}
                target_data = target_positions.get(symbol, {})
                target_qty = float(target_data.get('quantity', 0.0)) if target_data else 0.0
                
                diff = target_qty - current_qty
                
                # Apply Threshold (to avoid dust trading)
                if abs(diff) < 0.0001: # E.g., crypto dust
                    continue
                
                # 3. Create Order
                side = OrderSide.BUY if diff > 0 else OrderSide.SELL
                qty = abs(diff)
                
                # Get current price for Limit orders? 
                # For simplicity, using Market orders for now, or L3 decision could specify type.
                # Assuming Market Order by default unless specified.
                
                order = Order(
                    symbol=symbol,
                    quantity=qty, # Order model usually takes signed or unsigned with side?
                    # Based on Schema: quantity is float. Adapter handles side logic now.
                    # But for clarity in logs:
                    order_type=OrderType.MARKET,
                    side=side, # Schema might have this
                    status=OrderStatus.CREATED,
                    timestamp=datetime.now()
                )
                
                # 4. Execute
                try:
                    logger.info(f"Placing order: {side} {qty} {symbol}")
                    # Adapter place_order returns ID
                    # Pass signed quantity if Order object requires it, or Adapter handles it?
                    # Previous Adapter fix expects signed quantity in 'order.quantity'
                    # So we must set order.quantity to diff (signed)
                    order.quantity = diff 
                    
                    order_id = self.broker.place_order(order)
                    
                    if order_id:
                        self.active_orders[symbol] = order_id
                        logger.info(f"Order placed successfully: {order_id}. Tracking as active.")
                        state.execution_results.append({
                            "symbol": symbol,
                            "order_id": order_id,
                            "status": "submitted",
                            "diff": diff
                        })
                except Exception as e:
                    logger.error(f"Failed to place order for {symbol}: {e}")
                    state.execution_results.append({
                        "symbol": symbol,
                        "error": str(e),
                        "status": "failed"
                    })

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
    [Task FIX-CRIT-001] Execution Safety: Purchasing Power Check
    [Task P1-002] Deadlock Watchdog: Timeout & Auto-Recovery
    [Task P0-EXEC-01] Reserve Cash Mechanism
    [Task P1-EXEC-03] In-flight Order Status Management & Frozen Capital Tracking
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
        
        # 内部跟踪: symbol -> {'id': order_id, 'ts': timestamp, 'frozen_amount': float}
        # 修改结构以支持超时检查和资金冻结跟踪
        self.active_orders: Dict[str, Dict[str, Any]] = {} 
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
        [Fix] Ensures frozen capital is released using the correct key 'frozen_amount'.
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
                    current_active = self.active_orders.get(symbol)
                    
                    # Handle legacy string format or new dict format
                    current_id = current_active if isinstance(current_active, str) else (current_active.get('id') if current_active else None)

                    if current_id == str(order_id):
                        # [Fix] Retrieve frozen amount using correct key 'frozen_amount'
                        # Fallback to 'reserved' for backward compatibility during hot-reload
                        frozen_amt = 0.0
                        if isinstance(current_active, dict):
                             frozen_amt = current_active.get('frozen_amount', current_active.get('reserved', 0.0))
                        
                        # [Task P0-EXEC-01] Release Reservation on terminal state
                        if frozen_amt > 0 and self.tlm:
                             await self.tlm.release_cash(frozen_amt)

                        del self.active_orders[symbol]
                        logger.info(f"Order {order_id} for {symbol} finished ({event}). Released {frozen_amt}. Removed from active tracking.")
                    
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
                # [Task P1-EXEC-03] Watchdog & Ambiguous Order Handling
                if symbol in self.active_orders:
                    active_info = self.active_orders[symbol]
                    
                    # Migration/Safety: Handle if it's still a string
                    if isinstance(active_info, str):
                         active_info = {'id': active_info, 'ts': datetime.now().timestamp(), 'frozen_amount': 0.0}
                         self.active_orders[symbol] = active_info
                    
                    order_id = active_info.get('id')
                    ts = active_info.get('ts', datetime.now().timestamp())
                    # [Task P1-EXEC-03] Retrieve frozen amount
                    frozen_amt = active_info.get('frozen_amount', active_info.get('reserved', 0.0))
                    
                    # A. Ambiguous State Recovery (Timeout Recovery)
                    if order_id == 'AMBIGUOUS':
                        logger.warning(f"Checking status for ambiguous order: {symbol}")
                        try:
                            # Attempt to find the order via open orders scan
                            open_orders = self.broker.get_all_open_orders()
                            # Assuming get_all_open_orders returns list of Order objects
                            found_order = next((o for o in open_orders if o.symbol == symbol), None)
                            
                            if found_order:
                                logger.info(f"Recovered ambiguous order for {symbol}. ID: {found_order.id}")
                                active_info['id'] = found_order.id
                                active_info['ts'] = datetime.now().timestamp() # Reset timeout
                                self.active_orders[symbol] = active_info
                                continue # Now tracked normally
                            else:
                                # Not found. If timeout exceeded, assume failed and release.
                                if (datetime.now().timestamp() - ts) > 30: # 30s grace period for appearance
                                    logger.error(f"Ambiguous order for {symbol} not found after 30s. Assuming failed. Releasing {frozen_amt}.")
                                    if self.tlm and frozen_amt > 0:
                                        await self.tlm.release_cash(frozen_amt)
                                    del self.active_orders[symbol]
                                    # Proceed to place new order? No, wait next cycle to be safe.
                                else:
                                    logger.info(f"Ambiguous order {symbol} not yet found. Waiting...")
                        except Exception as e:
                            logger.error(f"Error checking ambiguous order {symbol}: {e}")
                        continue # Skip processing for this symbol
                    
                    # B. Standard Deadlock Watchdog (Timeout > 60s)
                    if (datetime.now().timestamp() - ts) > 60:
                        logger.warning(f"Order {order_id} for {symbol} stale (>60s). Checking status...")
                        try:
                            # Verify status with broker (Synchronous call expected)
                            status_res = self.broker.get_order_status(order_id)
                            
                            should_clear = False
                            if status_res.get("status") == "error":
                                logger.warning(f"Order {order_id} not found/error. Clearing active lock.")
                                should_clear = True
                            else:
                                data = status_res.get("data", {})
                                status = data.get("status", "").lower()
                                TERMINAL_STATES = {'filled', 'canceled', 'expired', 'rejected', 'stopped', 'done_for_day', 'replaced'}
                                if status in TERMINAL_STATES:
                                    logger.info(f"Stale order {order_id} is actually {status}. Clearing active lock.")
                                    should_clear = True
                            
                            if should_clear:
                                # [Task P1-EXEC-03] Ensure frozen amount is released if not already done
                                if self.tlm and frozen_amt > 0:
                                     await self.tlm.release_cash(frozen_amt)
                                del self.active_orders[symbol]
                                # Proceed to execute new order logic for this symbol
                            else:
                                logger.info(f"Order {order_id} still active. Keeping lock.")
                                continue
                        except Exception as e:
                            logger.error(f"Error checking stale order {order_id}: {e}")
                            continue # Fail safe, keep lock
                    else:
                        logger.info(f"Skipping {symbol}: Active order {order_id} pending.")
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
                
                # [Task P0-EXEC-01] Execution Safety with Reservation
                estimated_cost = 0.0
                reserved_success = False

                # [Task FIX-CRIT-001] Execution Safety: Check Purchasing Power for BUY orders
                if side == OrderSide.BUY:
                    estimated_price = 0.0
                    # Fix: Iterate over market_data_batch to find price
                    if hasattr(state, 'market_data_batch') and state.market_data_batch:
                        for md in state.market_data_batch:
                            # Robustly handle object or dict
                            md_sym = getattr(md, 'symbol', None) or (md.get('symbol') if isinstance(md, dict) else None)
                            if md_sym == symbol:
                                estimated_price = getattr(md, 'close', 0.0) or (md.get('close') if isinstance(md, dict) else 0.0)
                                break
                    
                    # Fail-Closed: If price is missing or invalid, do NOT proceed.
                    if estimated_price <= 0:
                        logger.error(f"OrderManager: Cannot verify funds. Missing/Invalid price for {symbol}. Skipping order.")
                        state.execution_results.append({
                            "symbol": symbol,
                            "error": "Missing Price for Fund Check",
                            "status": "skipped"
                        })
                        continue

                    estimated_cost = qty * estimated_price
                    
                    # Attempt Reservation
                    if self.tlm:
                        reserved_success = await self.tlm.reserve_cash(estimated_cost)
                        if not reserved_success:
                            logger.error(f"OrderManager: Insufficient funds (Reserved) for {symbol}. Cost: {estimated_cost}. Skipping.")
                            state.execution_results.append({"symbol": symbol, "error": "Insufficient Funds", "status": "skipped"})
                            continue

                # Get current price for Limit orders? 
                # For simplicity, using Market orders for now, or L3 decision could specify type.
                # Assuming Market Order by default unless specified.
                
                order = Order(
                    symbol=symbol,
                    quantity=diff, # Pass signed quantity if required by adapter
                    # quantity=qty, # Original code used absolute, but adapter likely expects signed diff or logic inside
                    order_type=OrderType.MARKET,
                    side=side,
                    status=OrderStatus.CREATED,
                    timestamp=datetime.now()
                )
                
                # 4. Execute
                try:
                    logger.info(f"Placing order: {side} {qty} {symbol}")
                    # Adapter place_order returns ID
                    # Ensure quantity matches what adapter expects (signed vs unsigned + side)
                    # Assuming adapter handles it based on Side + Qty
                    order.quantity = diff # [Correction] Ensure consistency
                    
                    order_id = self.broker.place_order(order)
                    
                    if order_id:
                        # [Task P1-EXEC-03] Store with frozen_amount
                        self.active_orders[symbol] = {
                            'id': order_id, 
                            'ts': datetime.now().timestamp(),
                            'frozen_amount': estimated_cost if reserved_success else 0.0
                        }
                        
                        logger.info(f"Order placed: {order_id}. Frozen: {estimated_cost if reserved_success else 0.0}")
                        state.execution_results.append({
                            "symbol": symbol,
                            "order_id": order_id,
                            "status": "submitted",
                            "diff": diff
                        })
                    else:
                        # Broker returned None/False immediately? Rollback.
                        if reserved_success and self.tlm:
                            await self.tlm.release_cash(estimated_cost)

                except Exception as e:
                    # [Task P1-EXEC-03] Network Timeout Handling -> AMBIGUOUS
                    error_msg = str(e).lower()
                    if any(x in error_msg for x in ['timeout', 'connection', '504', '502']):
                        logger.warning(f"Network timeout placing order for {symbol}. Entering AMBIGUOUS state. Error: {e}")
                        self.active_orders[symbol] = {
                            'id': 'AMBIGUOUS',
                            'ts': datetime.now().timestamp(),
                            'frozen_amount': estimated_cost if reserved_success else 0.0
                        }
                        # Do NOT release cash here; Watchdog will handle.
                        state.execution_results.append({
                            "symbol": symbol,
                            "error": "Ambiguous (Timeout)",
                            "status": "unknown"
                        })
                    else:
                        logger.error(f"Failed to place order for {symbol}: {e}")
                        # Rollback reservation on exception
                        if reserved_success and self.tlm:
                            await self.tlm.release_cash(estimated_cost)
                        
                        state.execution_results.append({
                            "symbol": symbol,
                            "error": str(e),
                            "status": "failed"
                        })

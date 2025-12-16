"""
Phoenix_project/execution/adapters.py
[Phase 3 Task 1] Fix Adapter Long/Short Inversion (Suicide Bug).
Explicitly pass 'side' and prioritize it over quantity sign inference.
[Phase 3 Task 2] Fix Adapter Hearing Loss (WebSocket & Watchdog).
1. Implement subscribe_order_status with unified stream handler.
2. Add Thread Watchdog to monitor and restart dead streams.
[Task 1.1] Added SimulatedBroker for Backtesting/Paper Trading isolation.
[Task P0-EXEC-02] Enforce Decimal Precision.
[Task P2-EXEC-04] Rate Limiting & Error Classification.
"""
import pandas as pd
import asyncio
import uuid
import logging
import os
import time
import threading # [Fix] Added for thread-safe RateLimiter
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from decimal import Decimal
from abc import ABC, abstractmethod

from Phoenix_project.monitor.logging import get_logger
from Phoenix_project.execution.interfaces import IBrokerAdapter
from Phoenix_project.core.schemas.data_schema import Order, OrderStatus, Fill, Position

# [任务 B.2] 导入 Alpaca 客户端
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest, LimitOrderRequest,
        StopOrderRequest, StopLimitOrderRequest, TrailingStopOrderRequest
    )
    from alpaca.trading.enums import OrderSide, TimeInForce
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.trading.stream import TradingStream
except ImportError:
    # 允许在没有安装 alpaca-py 的环境中运行回测
    pass

logger = get_logger(__name__)

# [Task P2-EXEC-04] Thread-Safe Sync Rate Limiter
class RateLimiter:
    """
    Thread-safe leaky bucket rate limiter for synchronous API calls.
    Ensures calls do not exceed the specified limit by sleeping the calling thread.
    """
    def __init__(self, calls_per_minute: int = 200):
        self.delay = 60.0 / calls_per_minute
        self.last_call = 0.0
        self._lock = threading.Lock()

    def wait_sync(self):
        """
        Blocks the calling thread until the rate limit allows a new request.
        """
        with self._lock:
            now = time.time()
            # Calculate when the next call is allowed
            target_time = self.last_call + self.delay
            wait_time = target_time - now
            
            if wait_time > 0:
                time.sleep(wait_time)
                self.last_call = time.time() # Update to actual time after sleep
            else:
                self.last_call = now

# --- 市场数据接口 ---

class AlpacaAdapter(IBrokerAdapter):
    """
    Adapter for Alpaca serving as both a market data provider
    and an execution broker.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the Alpaca adapter.
        """
        self.api_key = config.get('alpaca_api_key') or os.environ.get('ALPACA_API_KEY')
        self.api_secret = config.get('alpaca_api_secret') or os.environ.get('ALPACA_API_SECRET')
        self.paper = config.get('paper_trading', True)
        
        # [Task P2-EXEC-04] Rate Limiter Init (Sync & Thread-safe)
        self.rate_limiter = RateLimiter(calls_per_minute=190) # Slightly below 200 limit
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Alpaca API key/secret not provided or found in environment.")
            
        try:
            self.trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.api_secret,
                paper=self.paper
            )
            
            self.data_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.api_secret,
                raw_data=False 
            )
            
            self.trading_stream = TradingStream(
                api_key=self.api_key,
                secret_key=self.api_secret,
                paper=self.paper
            )
            self._stream_task = None
            
            # [Task 3.2] Callback Registries
            self._fill_handlers: List[Callable] = []
            self._status_handlers: List[Callable] = []
            self._watchdog_task: Optional[asyncio.Task] = None
            
            account = self.trading_client.get_account()
            logger.info(f"Alpaca connection successful. Account: {account.account_number}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}", exc_info=True)
            raise

    # [Task P2-EXEC-04] Error Classification Helper
    def _classify_error(self, e: Exception) -> str:
        msg = str(e).lower()
        # Network/Server errors -> RETRYABLE
        if any(x in msg for x in ['timeout', 'connection', '502', '503', '504', '429', 'rate limit']):
            return "RETRYABLE"
        # Client/Auth errors -> FATAL
        return "FATAL"

    # --- IMarketData 实现 ---
    def get_market_data(
        self, 
        symbols_list: List[str], 
        start_date: datetime, 
        end_date: datetime,
        timeframe_str: str = "1D"
    ) -> Dict[str, pd.DataFrame]:
        logger.info(f"Fetching Alpaca market data for {symbols_list} from {start_date} to {end_date}...")
        tf_map = { "1Min": TimeFrame.Minute, "1H": TimeFrame.Hour, "1D": TimeFrame.Day }
        alpaca_tf = tf_map.get(timeframe_str)
        if not alpaca_tf:
            logger.warning(f"Unsupported timeframe '{timeframe_str}'. Defaulting to '1D'.")
            alpaca_tf = TimeFrame.Day

        try:
            request_params = StockBarsRequest(
                symbol_or_symbols=symbols_list,
                timeframe=alpaca_tf,
                start=start_date,
                end=end_date
            )
            bars_response = self.data_client.get_stock_bars(request_params)
            bars_df = bars_response.df
            
            if bars_df.empty:
                logger.warning(f"Alpaca returned no data for symbols {symbols_list}.")
                return {}

            data_dict = {}
            if isinstance(bars_df.index, pd.MultiIndex):
                for symbol in bars_df.index.get_level_values('symbol').unique():
                    data_dict[symbol] = bars_df.loc[symbol]
            else:
                if symbols_list: data_dict[symbols_list[0]] = bars_df
            return data_dict

        except Exception as e:
            logger.error(f"Failed to get market data from Alpaca: {e}", exc_info=True)
            return {}

    # --- IExecutionBroker 实现 ---

    async def connect(self) -> None:
        """
        Starts the WebSocket stream in a background thread and initiates the Watchdog.
        """
        if self._stream_task and not self._stream_task.done():
            logger.info("Alpaca stream already running.")
            return
        
        # [Task 3.2] Unified Handler Registration
        self.trading_stream.subscribe_trade_updates(self._handle_stream_update)
        
        loop = asyncio.get_running_loop()
        # run() is blocking, so we execute it in the default executor (Thread)
        self._stream_task = loop.run_in_executor(None, self.trading_stream.run)
        logger.info("Alpaca Trading Stream connected (background).")
        
        # [Task 3.2] Start Watchdog
        if not self._watchdog_task or self._watchdog_task.done():
            self._watchdog_task = asyncio.create_task(self._monitor_stream())
            logger.info("Stream Watchdog started.")

    async def _monitor_stream(self):
        """
        [Task 3.2] Watchdog to monitor the stream thread and restart if it dies.
        """
        while True:
            try:
                if self._stream_task and self._stream_task.done():
                    logger.warning("Alpaca Trading Stream task ended unexpectedly. Attempting reconnection...")
                    
                    # Log exception if any
                    try:
                        exc = self._stream_task.exception()
                        if exc:
                            logger.error(f"Stream crashed with: {exc}")
                    except asyncio.CancelledError:
                        logger.info("Stream was cancelled.")
                        break # Stop watchdog if explicitly cancelled
                    except Exception:
                        pass

                    # Restart
                    loop = asyncio.get_running_loop()
                    self._stream_task = loop.run_in_executor(None, self.trading_stream.run)
                    logger.info("Alpaca Trading Stream restarted.")

                await asyncio.sleep(5) # Check every 5 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Watchdog error: {e}")
                await asyncio.sleep(5)

    async def _handle_stream_update(self, update: Any):
        """
        [Task 3.2] Unified Internal Handler for Stream Updates.
        Dispatches to registered callbacks.
        """
        try:
            event_type = getattr(update, 'event', None)
            # logger.debug(f"Stream Update Received: {event_type}")

            # 1. Dispatch Fills (fill, partial_fill)
            if event_type in ('fill', 'partial_fill'):
                for cb in self._fill_handlers:
                    try:
                        if asyncio.iscoroutinefunction(cb): await cb(update)
                        else: cb(update)
                    except Exception as e:
                        logger.error(f"Error in fill callback: {e}")

            # 2. Dispatch Order Status (new, canceled, rejected, etc.)
            for cb in self._status_handlers:
                try:
                    if asyncio.iscoroutinefunction(cb): await cb(update)
                    else: cb(update)
                except Exception as e:
                    logger.error(f"Error in status callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error handling stream update: {e}", exc_info=True)

    def disconnect(self) -> None:
        if self._watchdog_task:
            self._watchdog_task.cancel()
        pass

    def subscribe_fills(self, callback) -> None:
        """
        Registers a callback for trade fills.
        """
        if callback not in self._fill_handlers:
            self._fill_handlers.append(callback)
        logger.info("Subscribed to trade fills.")

    def subscribe_order_status(self, callback) -> None:
        """
        [Task 3.2] Registers a callback for order status updates.
        """
        if callback not in self._status_handlers:
            self._status_handlers.append(callback)
        logger.info("Subscribed to order status updates.")

    def get_all_open_orders(self) -> List[Order]:
        return []

    def get_portfolio_value(self) -> Decimal:
        try:
            # [Task P0-EXEC-02] Enforce Decimal precision
            val = self.trading_client.get_account().portfolio_value
            return Decimal(str(val)) if val is not None else Decimal("0.0")
        except Exception as e:
            logger.error(f"Error fetching portfolio value: {e}")
            return Decimal("0.0")

    def get_cash_balance(self) -> Decimal:
        try:
            # [Task P0-EXEC-02] Enforce Decimal precision
            val = self.trading_client.get_account().cash
            return Decimal(str(val)) if val is not None else Decimal("0.0")
        except Exception as e:
            logger.error(f"Error fetching cash balance: {e}")
            return Decimal("0.0")

    def get_position(self, symbol: str) -> Decimal:
        try:
            pos = self.trading_client.get_open_position(symbol)
            # [Task P0-EXEC-02] Enforce Decimal precision
            return Decimal(str(pos.qty))
        except Exception:
            return Decimal("0.0")

    def place_order(self, order: Order, price: Optional[float] = None) -> str:
        # Calculate side explicitly from signed quantity or Order attributes
        side = OrderSide.BUY if order.quantity > 0 else OrderSide.SELL
        
        order_data = {
            "symbol": order.symbol,
            "quantity": abs(order.quantity), 
            "side": side,                    
            "order_type": order.order_type.lower(),
            "limit_price": price if price is not None else order.limit_price
        }
        
        result = self.submit_order(order_data)
        if result.get("status") == "success":
            return str(result.get("order_id"))
        raise RuntimeError(f"Alpaca placement failed: {result.get('message')}")

    def submit_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # [Task P2-EXEC-04] Enforce Rate Limit (Thread-safe Wait)
            self.rate_limiter.wait_sync()

            symbol = order_data.get('symbol')
            qty = order_data.get('quantity')
            order_type = order_data.get('order_type', 'market')
            limit_price = order_data.get('limit_price')
            stop_price = order_data.get('stop_price')
            trail_percent = order_data.get('trail_percent')
            
            explicit_side = order_data.get('side')
            
            if not symbol or not qty:
                return {"status": "error", "message": "Symbol and Quantity are required.", "error_type": "FATAL"}

            if explicit_side:
                side = explicit_side
            else:
                side = OrderSide.BUY if qty > 0 else OrderSide.SELL

            abs_qty = abs(qty)

            if order_type == 'market':
                request = MarketOrderRequest(
                    symbol=symbol,
                    qty=abs_qty,
                    side=side,
                    time_in_force=TimeInForce.DAY
                )
            elif order_type == 'limit':
                if not limit_price:
                    return {"status": "error", "message": "Limit order requires 'limit_price'."}
                request = LimitOrderRequest(
                    symbol=symbol,
                    qty=abs_qty,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price
                )
            elif order_type == 'stop':
                if not stop_price:
                    return {"status": "error", "message": "Stop order requires 'stop_price'."}
                request = StopOrderRequest(
                    symbol=symbol,
                    qty=abs_qty,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    stop_price=stop_price
                )
            elif order_type == 'stop_limit':
                if not stop_price or not limit_price:
                    return {"status": "error", "message": "Stop-Limit order requires 'stop_price' and 'limit_price'."}
                request = StopLimitOrderRequest(
                    symbol=symbol,
                    qty=abs_qty,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    stop_price=stop_price,
                    limit_price=limit_price
                )
            elif order_type == 'trailing_stop':
                if not trail_percent:
                    return {"status": "error", "message": "Trailing stop requires 'trail_percent'."}
                request = TrailingStopOrderRequest(
                    symbol=symbol,
                    qty=abs_qty,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                    trail_percent=trail_percent
                )
            else:
                return {"status": "error", "message": f"Unsupported order type: {order_type}"}

            logger.info(f"Submitting {order_type} order to Alpaca: {side} {abs_qty} {symbol}")
            order = self.trading_client.submit_order(order_data=request)
            
            logger.info(f"Order submitted successfully. Order ID: {order.id}")
            return {"status": "success", "order_id": order.id, "data": order.dict()}

        except Exception as e:
            error_type = self._classify_error(e)
            logger.error(f"Failed to submit order to Alpaca ({error_type}): {e}", exc_info=True)
            return {
                "status": "error", 
                "message": str(e),
                "error_type": error_type # [Task P2-EXEC-04] Return classification
            }

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        try:
            order = self.trading_client.get_order_by_id(order_id)
            return {"status": "success", "data": order.dict()}
        except Exception as e:
            logger.error(f"Failed to get order status for {order_id}: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        try:
            self.trading_client.cancel_order_by_id(order_id)
            logger.info(f"Cancel request sent for order {order_id}.")
            return {"status": "success", "order_id": order_id}
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def get_account_info(self) -> Dict[str, Any]:
        try:
            account = self.trading_client.get_account()
            positions = self.trading_client.get_all_positions()
            return {
                "status": "success",
                "account": account.dict(),
                "positions": [p.dict() for p in positions]
            }
        except Exception as e:
            logger.error(f"Failed to get account info: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}


class SimulatedBroker(IBrokerAdapter):
    """
    In-memory simulated broker for backtesting and dry-runs.
    Maintains self.cash, self.positions, self.orders in memory.
    [Task 1.1] Implementation.
    """
    def __init__(self, config: Dict[str, Any]):
        self.initial_cash = Decimal(str(config.get("initial_cash", 100000.0)))
        self.cash = self.initial_cash
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        
        self._fill_handlers: List[Callable] = []
        self._status_handlers: List[Callable] = []
        self._logger = logger

    def connect(self) -> None:
        self._logger.info("SimulatedBroker connected (In-Memory).")

    def disconnect(self) -> None:
        self._logger.info("SimulatedBroker disconnected.")

    def subscribe_fills(self, callback: Callable) -> None:
        if callback not in self._fill_handlers:
            self._fill_handlers.append(callback)

    def subscribe_order_status(self, callback: Callable) -> None:
        if callback not in self._status_handlers:
            self._status_handlers.append(callback)

    def get_market_data(self, *args, **kwargs) -> Dict:
        return {}

    def get_portfolio_value(self) -> Decimal:
        # Estimated Total Value = Cash + Sum(Position Market Value)
        # [Task P0-EXEC-02] Enforce Decimal precision
        pos_val = sum(Decimal(str(p.market_value)) for p in self.positions.values())
        return self.cash + pos_val

    def get_cash_balance(self) -> Decimal:
        return self.cash

    def get_position(self, symbol: str) -> Decimal:
        pos = self.positions.get(symbol)
        # Position model quantity is float, so we cast back to Decimal for interface consistency
        return Decimal(str(pos.quantity)) if pos else Decimal("0.0")

    def get_positions(self) -> List[Position]:
        """
        Returns all open positions.
        Requested by blueprint to list all active holdings.
        """
        return list(self.positions.values())

    def get_all_open_orders(self) -> List[Order]:
        return [
            o for o in self.orders.values() 
            if o.status in [OrderStatus.NEW, OrderStatus.PENDING, OrderStatus.ACCEPTED, OrderStatus.PARTIALLY_FILLED]
        ]

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        order = self.orders.get(order_id)
        if order:
            return {"status": "success", "data": order.dict()}
        return {"status": "error", "message": "Order not found"}

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        order = self.orders.get(order_id)
        if order and order.status in [OrderStatus.NEW, OrderStatus.PENDING, OrderStatus.ACCEPTED]:
            order.status = OrderStatus.CANCELLED
            self._notify_status(order)
            return {"status": "success", "order_id": order_id}
        return {"status": "error", "message": "Order cannot be cancelled or not found"}

    def get_account_info(self) -> Dict[str, Any]:
        return {
            "status": "success",
            "account": {
                "cash": str(self.cash),
                "portfolio_value": str(self.get_portfolio_value()),
                "currency": "USD",
                "buying_power": str(self.cash)
            },
            "positions": [p.dict() for p in self.positions.values()]
        }

    def place_order(self, order: Order, price: Optional[float] = None) -> str:
        """
        Simulates order placement and immediate execution at 'price'.
        """
        if price is None:
            if order.limit_price:
                price = float(order.limit_price)
            else:
                raise ValueError("SimulatedBroker requires 'price' (or order.limit_price) for execution.")

        exec_price = Decimal(str(price))
        qty = Decimal(str(order.quantity)) # Signed quantity: +Buy, -Sell

        # 1. Update Order State
        order.status = OrderStatus.FILLED
        if not order.id:
            order.id = str(uuid.uuid4())
        self.orders[order.id] = order

        # 2. Update Cash
        cost = qty * exec_price
        self.cash -= cost

        # 3. Update Position
        self._update_position(order.symbol, qty, exec_price)

        # 4. Create Fill
        fill = Fill(
            id=str(uuid.uuid4()),
            order_id=order.id,
            symbol=order.symbol,
            timestamp=datetime.utcnow(),
            quantity=float(qty),
            price=float(exec_price),
            commission=Decimal("0.0")
        )

        # 5. Notify Callbacks
        self._notify_status(order)
        self._notify_fill(fill)

        return order.id

    def _update_position(self, symbol: str, qty: Decimal, price: Decimal):
        if symbol not in self.positions:
            if qty == 0: return
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=float(qty),
                average_price=float(price),
                market_value=float(qty * price),
                unrealized_pnl=0.0
            )
        else:
            pos = self.positions[symbol]
            current_qty = Decimal(str(pos.quantity))
            current_avg = Decimal(str(pos.average_price))
            
            new_qty = current_qty + qty
            
            if new_qty == 0:
                del self.positions[symbol]
            else:
                if (current_qty > 0 and qty > 0) or (current_qty < 0 and qty < 0):
                    current_cost = current_qty * current_avg
                    added_cost = qty * price
                    new_avg = (current_cost + added_cost) / new_qty
                    pos.average_price = float(new_avg)
                
                pos.quantity = float(new_qty)
                pos.market_value = float(new_qty * price)
                pos.unrealized_pnl = float((price - Decimal(str(pos.average_price))) * new_qty)

    def _notify_status(self, order: Order):
        for cb in self._status_handlers:
            try:
                if asyncio.iscoroutinefunction(cb):
                    asyncio.create_task(cb(order))
                else:
                    cb(order)
            except Exception as e:
                self._logger.error(f"Error in simulated status callback: {e}")

    def _notify_fill(self, fill: Fill):
        for cb in self._fill_handlers:
            try:
                if asyncio.iscoroutinefunction(cb):
                    asyncio.create_task(cb(fill))
                else:
                    cb(fill)
            except Exception as e:
                self._logger.error(f"Error in simulated fill callback: {e}")

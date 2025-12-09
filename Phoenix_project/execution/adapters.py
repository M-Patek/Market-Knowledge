"""
Phoenix_project/execution/adapters.py
[Phase 3 Task 1] Fix Adapter Long/Short Inversion (Suicide Bug).
Explicitly pass 'side' and prioritize it over quantity sign inference.
[Phase 3 Task 2] Fix Adapter Hearing Loss (WebSocket & Watchdog).
1. Implement subscribe_order_status with unified stream handler.
2. Add Thread Watchdog to monitor and restart dead streams.
"""
import pandas as pd
from typing import List, Dict, Any, Optional, Callable
from datetime import datetime
from abc import ABC, abstractmethod
import os
import asyncio

from Phoenix_project.monitor.logging import get_logger
from Phoenix_project.execution.interfaces import IBrokerAdapter
from Phoenix_project.core.schemas.data_schema import Order, OrderStatus

# [任务 B.2] 导入 Alpaca 客户端
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

logger = get_logger(__name__)

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

    # --- IMarketData 实现 ---
    # (保持原样)
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

    # (其余方法保持不变)
    def get_portfolio_value(self) -> float:
        try:
            return float(self.trading_client.get_account().portfolio_value)
        except Exception as e:
            logger.error(f"Error fetching portfolio value: {e}")
            return 0.0

    def get_cash_balance(self) -> float:
        try:
            return float(self.trading_client.get_account().cash)
        except Exception as e:
            logger.error(f"Error fetching cash balance: {e}")
            return 0.0

    def get_position(self, symbol: str) -> float:
        try:
            pos = self.trading_client.get_open_position(symbol)
            return float(pos.qty)
        except Exception:
            return 0.0

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
        symbol = order_data.get('symbol')
        qty = order_data.get('quantity')
        order_type = order_data.get('order_type', 'market')
        limit_price = order_data.get('limit_price')
        stop_price = order_data.get('stop_price')
        trail_percent = order_data.get('trail_percent')
        
        explicit_side = order_data.get('side')
        
        if not symbol or not qty:
            msg = "Order submission failed: Symbol and Quantity are required."
            logger.error(msg)
            return {"status": "error", "message": msg}

        if explicit_side:
            side = explicit_side
        else:
            side = OrderSide.BUY if qty > 0 else OrderSide.SELL

        abs_qty = abs(qty)

        try:
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
            logger.error(f"Failed to submit order to Alpaca: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

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

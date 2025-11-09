"""
券商适配器 (Broker Adapters)
实现 IBrokerAdapter 接口，用于连接模拟或真实的券商。
"""
from .interfaces import IBrokerAdapter, FillCallback, OrderStatusCallback
from typing import List, Optional, Dict, Any
from datetime import datetime
import time
import os
import asyncio
import threading
import pandas as pd # 修复 _convert_alpaca... 中 pd 未定义的错误

# 导入 Alpaca API (假设已通过 requirements.txt 安装)
try:
    import alpaca_trade_api as tradeapi
    # [第 1 步：添加 Imports]
    from alpaca_trade_api.stream import Stream # <-- [新] 导入 Stream
    from alpaca_trade_api.common import URL
except ImportError:
    print("Warning: 'alpaca-trade-api' not found. PaperTradingBrokerAdapter will not work.")
    tradeapi = None
    Stream = None
    URL = None

# [第 1 步：添加 Imports]
from threading import Thread # <-- [新] 导入 Thread


# FIX (E2, E4): 从核心模式导入 Order, Fill, OrderStatus
# 修正：将 'core.schemas...' 转换为 'Phoenix_project.core.schemas...'
from Phoenix_project.core.schemas.data_schema import Order, Fill, OrderStatus
# 导入 DataManager 用于 SimBroker 获取价格
# [阶段 1 变更]：不再需要，价格将通过 place_order 传入
# from Phoenix_project.data_manager import DataManager 
from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)


class SimulatedBrokerAdapter(IBrokerAdapter):
    """
    [阶段 1 实现]
    一个用于回测和开发的模拟券商。
    它在 'place_order' 被调用时立即模拟成交。
    """
    def __init__(self):
        self.open_orders: Dict[str, Order] = {} # key: order_id
        
        # 模拟交易成本
        self.slippage = 0.001 # 0.1%
        self.commission = 0.0 # 0 佣金
        
        self.fill_callback: Optional[FillCallback] = None
        self.order_status_callback: Optional[OrderStatusCallback] = None
        
        self.log_prefix = "SimBroker:"
        logger.info(f"{self.log_prefix} Initialized (Immediate Execution Mode).")

    def connect(self) -> None:
        logger.info(f"{self.log_prefix} Connection established.")
        pass

    def disconnect(self) -> None:
        logger.info(f"{self.log_prefix} Connection closed.")
        pass

    def subscribe_fills(self, callback: FillCallback) -> None:
        self.fill_callback = callback
        logger.info(f"{self.log_prefix} Fill callback subscribed.")

    def subscribe_order_status(self, callback: OrderStatusCallback) -> None:
        self.order_status_callback = callback
        logger.info(f"{self.log_prefix} Order status callback subscribed.")

    def place_order(self, order: Order, price: Optional[float] = None) -> str:
        """
        [阶段 1 实现]
        模拟订单执行。
        在回测中，此方法被调用时，我们假定订单立即以传入的 'price' 成交。
        """
        if order.id in self.open_orders:
            order.status = OrderStatus.REJECTED
            if self.order_status_callback:
                self.order_status_callback(order)
            logger.warning(f"{self.log_prefix} Duplicate order ID rejected: {order.id}")
            raise ValueError(f"Duplicate order ID: {order.id}")
        
        if price is None or price <= 0:
            order.status = OrderStatus.REJECTED
            if self.order_status_callback:
                self.order_status_callback(order)
            logger.error(f"{self.log_prefix} Order {order.id} rejected: No valid price provided for simulation.")
            raise ValueError(f"Simulation requires a valid price. Got {price} for {order.symbol}")

        logger.info(f"{self.log_prefix} Simulating order {order.id}: {order.quantity} @ {order.symbol} (Market Price: {price})")
        
        # 1. 模拟状态转换：NEW -> ACCEPTED
        order.status = OrderStatus.ACCEPTED
        self.open_orders[order.id] = order # 短暂添加
        if self.order_status_callback:
            self.order_status_callback(order)
            
        # 2. 模拟执行和滑点
        fill_price = price
        if order.quantity > 0: # Buy
            fill_price *= (1 + self.slippage)
        else: # Sell
            fill_price *= (1 - self.slippage)

        fill = Fill(
            id=f"fill-{order.id}",
            order_id=order.id,
            symbol=order.symbol,
            timestamp=datetime.utcnow(), # 在回测中，这应使用事件时间戳
            quantity=order.quantity,
            price=fill_price,
            commission=self.commission
        )
        
        # 3. 模拟状态转换：ACCEPTED -> FILLED
        order.status = OrderStatus.FILLED
        if self.order_status_callback:
            self.order_status_callback(order)
            
        # 4. 触发成交回调
        if self.fill_callback:
            self.fill_callback(fill)
        
        # 5. 从活动订单中移除
        if order.id in self.open_orders:
            del self.open_orders[order.id]
            
        return order.id

    def cancel_order(self, order_id: str) -> bool:
        # 在这个即时成交的模拟器中，订单永远不会停留在公开状态
        logger.warning(f"{self.log_prefix} Cancel called on {order_id}, but simulation is immediate.")
        return False

    def get_order_status(self, order_id: str) -> Optional[Order]:
        # 订单不会保持开放
        return self.open_orders.get(order_id)

    def get_all_open_orders(self) -> List[Order]:
        return list(self.open_orders.values())

    # --- 模拟器特有的方法 ---
    # 移除了 execute_order，因为逻辑已合并到 place_order

    # --- 模拟的 Getters ---
    def get_portfolio_value(self) -> float:
        logger.warning(f"{self.log_prefix} get_portfolio_value() is not implemented in SimBroker.")
        return 0.0

    def get_cash_balance(self) -> float:
        logger.warning(f"{self.log_prefix} get_cash_balance() is not implemented in SimBroker.")
        return 0.0

    def get_position(self, symbol: str) -> float:
        logger.warning(f"{self.log_prefix} get_position() is not implemented in SimBroker.")
        return 0.0
        
    def get_market_data(self, symbol: str, start: datetime, end: datetime) -> List[dict]:
        return []


class AlpacaAdapter(IBrokerAdapter):
    """
    [阶段 2 实现]
    连接到外部券商 (例如 Alpaca) 的虚拟交易 (Paper Trading) 端点。
    订单是异步发送的，成交是通过 WebSocket 监听器异步接收的。
    """
    def __init__(self, api_key: str, api_secret: str, paper_base_url: str):
        self.log_prefix = "PaperBroker(Alpaca):"
        if not tradeapi or not Stream or not URL:
            logger.critical(f"{self.log_prefix} 'alpaca-trade-api' library not installed. This adapter cannot function.")
            raise ImportError("Alpaca libraries not found.")
            
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = paper_base_url # e.g., "https://paper-api.alpaca.markets"
        
        self.api: Optional[tradeapi.REST] = None
        # self.conn: Optional[Stream] = None # [旧]
        
        self.fill_callback: Optional[FillCallback] = None
        self.order_status_callback: Optional[OrderStatusCallback] = None
        
        # [旧] Event loop logic
        # self.loop = asyncio.new_event_loop()
        # self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
        # self.thread.start()
        
        # --- [第 2 步：修改 __init__] ---
        self.stream: Optional[Stream] = None
        self.stream_thread: Optional[Thread] = None
        self._stream_running = False # [任务 4] 添加标志
        # --- [添加结束] ---
        
        logger.info(f"{self.log_prefix} Initialized.") # [旧] 移除了 "Async event loop started."

    # [旧] Event loop logic
    # def _run_event_loop(self):
    #     """在专用线程中运行 asyncio 事件循环。"""
    #     logger.info(f"{self.log_prefix} Async event loop running in thread.")
    #     asyncio.set_event_loop(self.loop)
    #     try:
    #         self.loop.run_forever()
    #     finally:
    #         self.loop.close()
    #         logger.info(f"{self.log_prefix} Async event loop stopped.")

    def connect(self) -> None:
        """建立与 REST API 和 WebSocket Stream 的连接。"""
        try:
            self.api = tradeapi.REST(self.api_key, self.api_secret, URL(self.base_url))
            account = self.api.get_account()
            logger.info(f"{self.log_prefix} REST API connection successful. Account: {account.account_number} (Paper: {account.paper_trading})")
            
            # --- [第 3 步：实现 connect] ---
            # 启动 WebSocket 监听器
            # asyncio.run_coroutine_threadsafe(self._start_websocket_listener(), self.loop) # [旧]
            
            logger.info(f"{self.log_prefix} Connecting to Alpaca WebSocket stream...")
            self.stream = Stream(
                key_id=self.api_key,
                secret_key=self.api_secret,
                base_url=URL(self.base_url) # [修复] 确保使用 URL() 包装
            )
            
            # 订阅 Alpaca 的“交易更新”频道
            # 关键：将它连接到我们已经实现了的 _on_trade_update 方法
            # [修复] _on_trade_update 是一个异步方法，需要一个事件循环
            # 我们需要将回调包装起来，以便在我们的 loop (如果存在) 或新循环中运行
            
            # 简易修复：直接订阅同步回调
            # （注意：Alpaca-trade-api v2+ 的 Stream.run() 是同步阻塞的，
            # 它会在自己的线程中处理异步回调）
            
            # 修正：Alpaca-trade-api Stream v1.x (根据 requirements.txt) 
            # 期望的是同步回调。我们需要将异步逻辑包装在同步函数中。
            
            # 让我们重新审视 _start_websocket_listener 的逻辑。
            # Alpaca V1 (alpaca-trade-api-python) 的 Stream.run() 是同步阻塞的。
            # 回调 (on) 应该是异步函数 (async def)。
            # 这意味着 self.conn.run() 必须在一个单独的线程中运行，
            # 并且它会管理自己的事件循环来调用异步回调。

            # 让我们遵循原始计划，它似乎是为 V1 设计的。
            
            self.stream.subscribe_trade_updates(self._on_trade_update) # 订阅异步回调

            # 将 self.stream.run() 放入一个单独的线程，因为它是一个阻塞操作
            # daemon=True 意味着如果主程序退出，这个线程也会退出
            self._stream_running = True # [任务 4] 设置标志
            self.stream_thread = Thread(target=self._run_stream, daemon=True)
            self.stream_thread.start()
            
            logger.info(f"{self.log_prefix} Alpaca WebSocket stream thread started.")
            
            # --- [第 3 步结束] ---
            
        except Exception as e:
            logger.error(f"{self.log_prefix} Connection failed: {e}", exc_info=True)
            raise

    # --- [第 3 步：实现 _run_stream] ---
    def _run_stream(self) -> None:
        """
        [新] 在线程中运行流的辅助方法，包含错误处理。
        [任务 4 已修改] 添加自动重连循环。
        """
        # [任务 4] 循环，直到 _stream_running 为 False
        while self._stream_running:
            try:
                if self.stream:
                    logger.info(f"{self.log_prefix} WebSocket stream run() loop starting...")
                    self.stream.run() # 这是阻塞的
                    
                    # 如果 run() 正常退出 (例如被 self.stream.stop() 调用)
                    if not self._stream_running:
                        logger.info(f"{self.log_prefix} WebSocket stream run() loop finished cleanly.")
                        break # 退出 while 循环
                    else:
                        logger.warning(f"{self.log_prefix} WebSocket stream exited unexpectedly. Reconnecting...")
                else:
                    logger.error(f"{self.log_prefix} Stream object is None. Cannot run.")
                    
            except Exception as e:
                logger.error(f"{self.log_prefix} Alpaca WebSocket stream crashed: {e}", exc_info=True)
            
            if self._stream_running:
                # [任务 4] 发生崩溃或意外退出，等待 5 秒后重试
                logger.info(f"{self.log_prefix} Waiting 5 seconds before reconnecting WebSocket...")
                time.sleep(5) 
                
                # [任务 4] 尝试重建流对象
                try:
                    logger.info(f"{self.log_prefix} Re-initializing stream...")
                    self.stream = Stream(
                        key_id=self.api_key,
                        secret_key=self.api_secret,
                        base_url=URL(self.base_url)
                    )
                    self.stream.subscribe_trade_updates(self._on_trade_update)
                except Exception as e:
                     logger.error(f"{self.log_prefix} Failed to re-initialize stream, retrying in 5s: {e}")
                     time.sleep(5) # 再次睡眠
    # --- [第 3 步结束] ---


    async def _on_trade_update(self, data): # [修复] 保持为 async，Stream.run() 会处理它
        """
        [重命名]
        处理来自 Alpaca 的订单更新和成交。
        (原为 _start_websocket_listener 内部的 on_trade_update)
        """
        logger.debug(f"{self.log_prefix} WebSocket Data: {data}")
        
        # Alpaca 的 'trade_update' 事件包含 'order' 和 (如果成交) 'fill'
        event = data.event
        alpaca_order_data = data.order # 这是原始字典
        
        # 1. 转换订单状态
        try:
            order = self._convert_alpaca_order_to_order(alpaca_order_data)
            if self.order_status_callback:
                # [修复] 回调是同步的，不能在异步函数中 await
                # (假设回调是线程安全的或非阻塞的)
                self.order_status_callback(order)
        except Exception as e:
            logger.error(f"{self.log_prefix} Error converting Alpaca order: {e}", exc_info=True)

        # 2. 如果是 'fill' 或 'partial_fill'，转换 Fill
        if event == 'fill' or event == 'partial_fill':
            try:
                fill = self._convert_alpaca_trade_to_fill(data, alpaca_order_data) # 'data' 本身是 trade
                if self.fill_callback:
                    self.fill_callback(fill)
            except Exception as e:
                logger.error(f"{self.log_prefix} Error converting Alpaca fill: {e}", exc_info=True)


    def disconnect(self) -> None:
        # [旧]
        # if self.conn:
        #     asyncio.run_coroutine_threadsafe(self.conn.close(), self.loop)
        # if self.loop.is_running():
        #     self.loop.call_soon_threadsafe(self.loop.stop)
        # self.thread.join(timeout=5)
        # logger.info(f"{self.log_prefix} Connection closed.")
        
        # --- [第 4 步：实现 disconnect] ---
        try:
            self._stream_running = False # [任务 4] 设置标志，停止 _run_stream 循环
            if self.stream:
                logger.info(f"{self.log_prefix} Disconnecting Alpaca WebSocket stream...")
                self.stream.stop() # 告诉 WebSocket 停止 (V1 API)
            
            if self.stream_thread and self.stream_thread.is_alive():
                logger.info(f"{self.log_prefix} Waiting for stream thread to join...")
                self.stream_thread.join(timeout=5.0) # 等待线程结束
                
            logger.info(f"{self.log_prefix} Alpaca WebSocket stream disconnected.")
        except Exception as e:
            logger.error(f"{self.log_prefix} Error during Alpaca stream disconnection: {e}", exc_info=True)
        # --- [第 4 步结束] ---
        self.api = None

    def subscribe_fills(self, callback: FillCallback) -> None:
        self.fill_callback = callback
        logger.info(f"{self.log_prefix} Fill callback subscribed.")

    def subscribe_order_status(self, callback: OrderStatusCallback) -> None:
        self.order_status_callback = callback
        logger.info(f"{self.log_prefix} Order status callback subscribed.")

    def place_order(self, order: Order, price: Optional[float] = None) -> str:
        """
        [阶段 2 实现]
        异步提交订单。忽略 'price' 参数。
        """
        if not self.api:
            raise ConnectionError(f"{self.log_prefix} API not connected. Call connect() first.")
            
        logger.info(f"{self.log_prefix} Submitting order {order.id} for {order.symbol}")
        
        # [修复] 原始代码中没有 self.loop。
        # _async_place_order 是一个 async def，必须在事件循环中运行。
        # 但是，我们没有在 __init__ 中启动循环。
        
        # 简易修复：在专用于 Alpaca 的线程中运行阻塞调用。
        # （或者，如果 self.api.submit_order 是阻塞的，我们可以直接调用它，
        # 但这会阻塞调用者（OrderManager），这可能是不希望的）
        
        # 让我们使用一个新线程来防止阻塞 OrderManager
        def _submit_order_thread():
            try:
                alpaca_order = self.api.submit_order(
                    symbol=order.symbol,
                    qty=abs(order.quantity),
                    side="buy" if order.quantity > 0 else "sell",
                    type=order.order_type.lower(),
                    time_in_force=order.time_in_force.lower(),
                    limit_price=order.limit_price,
                    client_order_id=order.id # 使用我们的ID作为 client_order_id
                )
                logger.info(f"{self.log_prefix} Order submitted successfully. Broker ID: {alpaca_order.id}")
            except Exception as e:
                logger.error(f"{self.log_prefix} Failed to place order {order.id}: {e}", exc_info=True)
                # (可以触发一个 REJECTED 回调)
                order.status = OrderStatus.REJECTED
                if self.order_status_callback:
                    self.order_status_callback(order)

        # 在一个单独的线程中启动阻塞 API 调用
        submit_thread = Thread(target=_submit_order_thread, daemon=True)
        submit_thread.start()
        
        # 立即返回我们系统的订单ID
        return order.id

    # [旧]
    # async def _async_place_order(self, order: Order):
    #     """在事件循环中执行实际的 API 调用。"""
    #     ... (逻辑移至上面的 _submit_order_thread)

    # --- Alpaca 辅助转换函数 ---

    def _convert_alpaca_order_to_order(self, alpaca_order: Dict[str, Any]) -> Order:
        """将 Alpaca Order 字典转换为我们的 Order Pydantic 模型。"""
        
        # Alpaca 状态映射
        status_map = {
            "new": OrderStatus.ACCEPTED,
            "partially_filled": OrderStatus.PARTIALLY_FILLED,
            "filled": OrderStatus.FILLED,
            "done_for_day": OrderStatus.ACCEPTED, # (假设未成交的当日订单)
            "canceled": OrderStatus.CANCELLED,
            "expired": OrderStatus.CANCELLED,
            "rejected": OrderStatus.REJECTED,
            "pending_new": OrderStatus.PENDING,
            "pending_cancel": OrderStatus.PENDING,
            "pending_replace": OrderStatus.PENDING,
            "accepted": OrderStatus.ACCEPTED,
            "stopped": OrderStatus.CANCELLED, # (假设是止损单)
            "suspended": OrderStatus.PENDING,
            "calculated": OrderStatus.PENDING,
        }
        our_status = status_map.get(alpaca_order.get('status'), OrderStatus.PENDING)
        
        quantity = float(alpaca_order.get('qty', 0))
        if alpaca_order.get('side') == 'sell':
            quantity = -quantity
            
        created_at_str = alpaca_order.get('created_at')
        created_at_dt = datetime.utcnow() # 默认
        if created_at_str:
            try:
                # 尝试解析带 'Z' 的 ISO 格式
                created_at_dt = pd.to_datetime(created_at_str).to_pydatetime()
            except Exception:
                 logger.warning(f"{self.log_prefix} Could not parse Alpaca timestamp {created_at_str}")

        return Order(
            id=alpaca_order.get('client_order_id'), # 我们使用 client_order_id 作为我们的主键
            client_order_id=alpaca_order.get('id'), # 存储 Alpaca 的 ID
            symbol=alpaca_order.get('symbol'),
            quantity=quantity,
            order_type=alpaca_order.get('type', 'market').upper(),
            limit_price=float(alpaca_order.get('limit_price')) if alpaca_order.get('limit_price') else None,
            time_in_force=alpaca_order.get('time_in_force', 'gtc').upper(),
            status=our_status,
            created_at=created_at_dt
        )

    def _convert_alpaca_trade_to_fill(self, alpaca_trade: Any, alpaca_order_data: Dict[str, Any]) -> Fill:
        """将 Alpaca Trade (来自 WebSocket) 转换为我们的 Fill Pydantic 模型。"""
        
        # 'alpaca_trade' (data) 包含 'order' 字典
        # alpaca_order_data = alpaca_trade.order

        quantity = float(alpaca_trade.qty)
        if alpaca_order_data.get('side') == 'sell':
            quantity = -quantity
            
        timestamp_str = alpaca_trade.timestamp
        timestamp_dt = datetime.utcnow()
        if timestamp_str:
            try:
                timestamp_dt = pd.to_datetime(timestamp_str).to_pydatetime()
            except Exception:
                 logger.warning(f"{self.log_prefix} Could not parse Alpaca fill timestamp {timestamp_str}")


        return Fill(
            id=alpaca_trade.id, # 使用 Alpaca 的 trade ID
            order_id=alpaca_order_data.get('client_order_id'), # 链接到我们的 client_order_id
            symbol=alpaca_trade.symbol,
            timestamp=timestamp_dt,
            quantity=quantity,
            price=float(alpaca_trade.price),
            commission=float(alpaca_order_data.get('commission', 0.0)) # (注意：这可能是总佣金)
        )

    # --- 接口的其余部分 (TODO: 实现) ---
    def cancel_order(self, order_id: str) -> bool:
        # [任务 3 已实现]
        if not self.api:
            logger.error(f"{self.log_prefix} Cannot cancel order: API not connected.")
            return False
        
        logger.info(f"{self.log_prefix} Attempting to cancel order {order_id}...")
        try:
            # 'order_id' 是我们的 client_order_id。
            # 我们必须先获取 Alpaca 的 broker ID。
            order_data = self.api.get_order_by_client_order_id(order_id)
            broker_id = order_data.id
            self.api.cancel_order(broker_id)
            logger.info(f"{self.log_prefix} Cancel request for {order_id} (Broker ID: {broker_id}) successful.")
            return True
        except Exception as e:
            logger.error(f"{self.log_prefix} Failed to cancel order {order_id}: {e}", exc_info=True)
            return False

    def get_order_status(self, order_id: str) -> Optional[Order]:
        # [任务 3 已实现]
        if not self.api:
            logger.error(f"{self.log_prefix} Cannot get order status: API not connected.")
            return None
        
        try:
            # 'order_id' 是我们的 client_order_id
            alpaca_order = self.api.get_order_by_client_order_id(order_id)
            # alpaca_order 是一个 Httpx-Entity 对象, 将其转为 dict
            return self._convert_alpaca_order_to_order(alpaca_order.__dict__)
        except tradeapi.rest.APIError as e:
            if "order not found" in str(e).lower():
                logger.warning(f"{self.log_prefix} Order {order_id} not found via API.")
            else:
                logger.error(f"{self.log_prefix} API error getting order {order_id}: {e}", exc_info=True)
            return None
        except Exception as e:
            logger.error(f"{self.log_prefix} Failed to get order status for {order_id}: {e}", exc_info=True)
            return None

    def get_all_open_orders(self) -> List[Order]:
        # [任务 3 已实现]
        if not self.api:
            logger.error(f"{self.log_prefix} Cannot get open orders: API not connected.")
            return []
        
        try:
            alpaca_orders = self.api.list_orders(status='open')
            orders = [self._convert_alpaca_order_to_order(o.__dict__) for o in alpaca_orders]
            logger.info(f"{self.log_prefix} Fetched {len(orders)} open orders.")
            return orders
        except Exception as e:
            logger.error(f"{self.log_prefix} Failed to get all open orders: {e}", exc_info=True)
            return []

    def get_portfolio_value(self) -> float:
        try:
            if not self.api: return 0.0
            return float(self.api.get_account().portfolio_value)
        except Exception as e:
            logger.error(f"{self.log_prefix} get_portfolio_value() failed: {e}")
            return 0.0

    def get_cash_balance(self) -> float:
        try:
            if not self.api: return 0.0
            return float(self.api.get_account().cash)
        except Exception as e:
            logger.error(f"{self.log_prefix} get_cash_balance() failed: {e}")
            return 0.0

    def get_position(self, symbol: str) -> float:
        try:
            if not self.api: return 0.0
            return float(self.api.get_position(symbol).qty)
        except Exception: # (例如 alpaca_trade_api.rest.APIError: position not found)
            return 0.0
            
    def get_market_data(self, symbol: str, start: datetime, end: datetime) -> List[dict]:
        logger.warning(f"{self.log_prefix} get_market_data() not implemented.")
        return []

# 重命名 AlpacaAdapter 以匹配阶段 2 的计划
PaperTradingBrokerAdapter = AlpacaAdapter

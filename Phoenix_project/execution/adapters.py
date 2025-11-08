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
    from alpaca_trade_api.stream import Stream
    from alpaca_trade_api.common import URL
except ImportError:
    print("Warning: 'alpaca-trade-api' not found. PaperTradingBrokerAdapter will not work.")
    tradeapi = None
    Stream = None
    URL = None


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
        self.conn: Optional[Stream] = None
        
        self.fill_callback: Optional[FillCallback] = None
        self.order_status_callback: Optional[OrderStatusCallback] = None
        
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.thread.start()
        
        logger.info(f"{self.log_prefix} Initialized. Async event loop started.")

    def _run_event_loop(self):
        """在专用线程中运行 asyncio 事件循环。"""
        logger.info(f"{self.log_prefix} Async event loop running in thread.")
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_forever()
        finally:
            self.loop.close()
            logger.info(f"{self.log_prefix} Async event loop stopped.")

    def connect(self) -> None:
        """建立与 REST API 和 WebSocket Stream 的连接。"""
        try:
            self.api = tradeapi.REST(self.api_key, self.api_secret, URL(self.base_url))
            account = self.api.get_account()
            logger.info(f"{self.log_prefix} REST API connection successful. Account: {account.account_number} (Paper: {account.paper_trading})")
            
            # 启动 WebSocket 监听器
            asyncio.run_coroutine_threadsafe(self._start_websocket_listener(), self.loop)
            
        except Exception as e:
            logger.error(f"{self.log_prefix} Connection failed: {e}", exc_info=True)
            raise

    async def _start_websocket_listener(self):
        """在异步循环中初始化并运行 WebSocket 监听器。"""
        try:
            logger.info(f"{self.log_prefix} Starting WebSocket listener...")
            self.conn = Stream(
                self.api_key,
                self.api_secret,
                base_url=URL(self.base_url),
                data_feed='iex' # 仅用于交易流
            )

            # 注册处理函数 (trade_updates 包含 fills 和 order status)
            @self.conn.on(r'trade_updates')
            async def on_trade_update(data):
                """处理来自 Alpaca 的订单更新和成交。"""
                logger.debug(f"{self.log_prefix} WebSocket Data: {data}")
                
                # Alpaca 的 'trade_update' 事件包含 'order' 和 (如果成交) 'fill'
                event = data.event
                alpaca_order_data = data.order # 这是原始字典
                
                # 1. 转换订单状态
                try:
                    order = self._convert_alpaca_order_to_order(alpaca_order_data)
                    if self.order_status_callback:
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

            # 订阅交易更新
            await self.conn.subscribe_trade_updates()
            logger.info(f"{self.log_prefix} WebSocket listener running.")
            
        except Exception as e:
            logger.error(f"{self.log_prefix} Failed to start WebSocket listener: {e}", exc_info=True)

    def disconnect(self) -> None:
        if self.conn:
            asyncio.run_coroutine_threadsafe(self.conn.close(), self.loop)
        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=5)
        logger.info(f"{self.log_prefix} Connection closed.")
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
        
        # 将此阻塞I/O调用安排到异步循环中
        asyncio.run_coroutine_threadsafe(
            self._async_place_order(order),
            self.loop
        )
        
        # 立即返回我们系统的订单ID
        return order.id

    async def _async_place_order(self, order: Order):
        """在事件循环中执行实际的 API 调用。"""
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
        logger.warning(f"{self.log_prefix} cancel_order() not implemented.")
        # asyncio.run_coroutine_threadsafe(self.api.cancel_order(order_id), self.loop)
        return False

    def get_order_status(self, order_id: str) -> Optional[Order]:
        logger.warning(f"{self.log_prefix} get_order_status() not implemented.")
        # order = self.api.get_order_by_client_order_id(order_id)
        # return self._convert_alpaca_order_to_order(order)
        return None

    def get_all_open_orders(self) -> List[Order]:
        logger.warning(f"{self.log_prefix} get_all_open_orders() not implemented.")
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

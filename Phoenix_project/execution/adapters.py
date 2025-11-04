"""
券商适配器 (Broker Adapters)
实现 IBrokerAdapter 接口，用于连接模拟或真实的券商。
"""
from .interfaces import IBrokerAdapter, FillCallback, OrderStatusCallback
from typing import List, Optional, Dict, Any
from datetime import datetime
import time

# FIX (E2, E4): 从核心模式导入 Order, Fill, OrderStatus
# 修正：将 'core.schemas...' 转换为 'Phoenix_project.core.schemas...'
from Phoenix_project.core.schemas.data_schema import Order, Fill, OrderStatus


class SimulatedBrokerAdapter(IBrokerAdapter):
    """
    一个用于回测和开发的模拟券商。
    它模拟订单执行、成交和滑点。
    """
    def __init__(self, initial_cash: float = 1000000.0, slippage: float = 0.001, commission: float = 0.0):
        self.cash = initial_cash
        self.positions: Dict[str, float] = {} # key: symbol, value: quantity
        self.open_orders: Dict[str, Order] = {} # key: order_id
        
        self.slippage = slippage
        self.commission = commission
        
        self.fill_callback: Optional[FillCallback] = None
        self.order_status_callback: Optional[OrderStatusCallback] = None
        
        self.log_prefix = "SimBroker:"
        print(f"{self.log_prefix} Initialized with cash {initial_cash}")

    def connect(self) -> None:
        print(f"{self.log_prefix} Connection established.")
        pass

    def disconnect(self) -> None:
        print(f"{self.log_prefix} Connection closed.")
        pass

    def subscribe_fills(self, callback: FillCallback) -> None:
        self.fill_callback = callback
        print(f"{self.log_prefix} Fill callback subscribed.")

    def subscribe_order_status(self, callback: OrderStatusCallback) -> None:
        self.order_status_callback = callback
        print(f"{self.log_prefix} Order status callback subscribed.")

    def place_order(self, order: Order) -> str:
        if order.id in self.open_orders:
            order.status = OrderStatus.REJECTED
            if self.order_status_callback:
                self.order_status_callback(order)
            raise ValueError(f"Duplicate order ID: {order.id}")
            
        print(f"{self.log_prefix} Placing order {order.id}: {order.quantity} @ {order.symbol}")
        
        order.status = OrderStatus.ACCEPTED
        self.open_orders[order.id] = order
        
        if self.order_status_callback:
            self.order_status_callback(order)
            
        # 模拟器：立即尝试执行市价单
        if order.order_type == "MARKET":
            # 在真实系统中，执行是异步的。
            # 这里我们假设我们有一些市场数据来执行它。
            # (在回测中，这通常由回测引擎在下一个tick触发)
            pass
            
        return order.id

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self.open_orders:
            order = self.open_orders.pop(order_id)
            order.status = OrderStatus.CANCELLED
            print(f"{self.log_prefix} Order {order_id} cancelled.")
            if self.order_status_callback:
                self.order_status_callback(order)
            return True
        return False

    def get_order_status(self, order_id: str) -> Optional[Order]:
        return self.open_orders.get(order_id)

    def get_all_open_orders(self) -> List[Order]:
        return list(self.open_orders.values())

    def get_portfolio_value(self) -> float:
        # 简化的实现：只返回现金
        # 真实实现需要市价
        return self.cash 

    def get_cash_balance(self) -> float:
        return self.cash

    def get_position(self, symbol: str) -> float:
        return self.positions.get(symbol, 0.0)

    # --- 模拟器特有的方法 ---
    
    def execute_order(self, order_id: str, market_price: float):
        """
        由回测引擎调用，以特定的市场价格执行订单。
        """
        if order_id not in self.open_orders:
            return
            
        order = self.open_orders.pop(order_id)
        
        # 模拟滑点
        if order.quantity > 0: # Buy
            exec_price = market_price * (1 + self.slippage)
        else: # Sell
            exec_price = market_price * (1 - self.slippage)
            
        exec_cost = exec_price * order.quantity
        exec_commission = abs(exec_cost) * self.commission
        
        # 更新状态
        self.cash -= (exec_cost + exec_commission)
        self.positions[order.symbol] = self.positions.get(order.symbol, 0.0) + order.quantity
        
        order.status = OrderStatus.FILLED
        
        # FIX (E2): 创建 Fill 对象
        fill = Fill(
            id=f"fill_{order_id}",
            order_id=order_id,
            symbol=order.symbol,
            timestamp=datetime.utcnow(), # 实际应使用市场数据时间戳
            quantity=order.quantity,
            price=exec_price,
            commission=exec_commission
        )
        
        print(f"{self.log_prefix} Order {order_id} FILLED: {fill.quantity} @ {fill.price}")

        # 触发回调
        if self.order_status_callback:
            self.order_status_callback(order)
        if self.fill_callback:
            self.fill_callback(fill)


# FIX (E6): 移除了 AlpacaAdapter 的导入，添加一个占位符
class LiveBrokerAdapter(IBrokerAdapter):
    """
    (占位符) 真实券商 (如 Alpaca) 的适配器。
    """
    def __init__(self, api_key: str, api_secret: str, base_url: str):
        raise NotImplementedError("LiveBrokerAdapter is not implemented")

    def connect(self) -> None:
        raise NotImplementedError
    
    # ... (实现所有 IBrokerAdapter 方法) ...

}

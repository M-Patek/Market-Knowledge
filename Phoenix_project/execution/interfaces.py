"""
Execution Interfaces (Contracts)
- 定义执行层中组件之间交互的抽象基类 (ABC) 和数据模型。
"""
from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Optional, List
from enum import Enum

# --- 数据模型 ---

class Order(BaseModel):
    """定义一个订单对象"""
    ticker: str
    qty: float
    side: str  # 'buy' or 'sell'
    order_type: str  # 'market', 'limit'
    limit_price: Optional[float] = None
    broker_order_id: Optional[str] = None # 由 broker 返回的 ID
    internal_order_id: Optional[str] = None

class Fill(BaseModel):
    """定义一个订单成交或更新对象"""
    ticker: str
    order_id: str
    status: str # (应使用 OrderStatus 枚举)
    filled_qty: float
    avg_fill_price: float

# --- 关键修正 (Error 6): 添加缺失的定义 ---

class OrderStatus(Enum):
    """
    定义订单状态的标准枚举 (合约)。
    """
    NEW = "NEW"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    PENDING_CANCEL = "PENDING_CANCEL"
    ERROR = "ERROR"

class ExecutionAdapter(ABC):
    """
    定义执行适配器 (e.g., Alpaca, IBKR) 必须实现的接口 (合约)。
    """
    @abstractmethod
    async def place_order(self, order: Order) -> str:
        """
        发送订单到 broker。
        :return: Broker-side 订单 ID
        """
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """
        取消一个活动订单。
        :return: True if successful
        """
        pass

    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """获取订单的当前状态"""
        pass
        
    @abstractmethod
    async def get_positions(self) -> dict:
        """获取当前持仓"""
        pass
        
    @abstractmethod
    async def get_total_equity(self) -> float:
        """获取账户总资产净值"""
        pass
        
    @abstractmethod
    async def get_last_price(self, ticker: str) -> float:
        """获取标的的最新价格"""
        pass

# --- 现有接口 ---

class IBrokerAdapter(ExecutionAdapter):
    """
    (此接口现在是多余的，因为 ExecutionAdapter 已经定义了合约)
    (可以保留 IBrokerAdapter 并使其继承 ExecutionAdapter)
    """
    pass

class IOrderManager(ABC):
    """
    定义 Order Manager 的接口。
    """
    @abstractmethod
    async def generate_orders_from_signal(self, signal):
        """根据信号生成订单"""
        pass

    @abstractmethod
    async def on_fill(self, fill_event: Fill):
        """处理成交事件"""
        pass

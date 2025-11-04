"""
定义执行层（券商）的抽象接口。
"""
from abc import ABC, abstractmethod
from typing import List, Callable, Optional
from datetime import datetime

# FIX (E4): 从此文件移除 Order, Fill, OrderStatus 的Pydantic定义。
# 它们现在是核心模式的一部分，并从 data_schema 导入。
# 修正：将 'core.schemas...' 转换为 'Phoenix_project.core.schemas...'
from Phoenix_project.core.schemas.data_schema import Order, Fill, OrderStatus

# 定义回调函数类型
FillCallback = Callable[[Fill], None]
OrderStatusCallback = Callable[[Order], None]


class IBrokerAdapter(ABC):
    """
    券商适配器接口 (Interface)。
    定义了与任何券商（模拟或真实）交互所需的所有标准方法。
    """

    @abstractmethod
    def connect(self) -> None:
        """
        建立与券商API的连接。
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """
        断开与券商API的连接。
        """
        pass

    @abstractmethod
    def subscribe_fills(self, callback: FillCallback) -> None:
        """
        注册一个回调函数，以便在收到成交回报时调用。
        """
        pass
    
    @abstractmethod
    def subscribe_order_status(self, callback: OrderStatusCallback) -> None:
        """
        注册一个回调函数，以便在订单状态更新时调用。
        """
        pass

    @abstractmethod
    def place_order(self, order: Order) -> str:
        """
        向券商提交一个新订单。
        :param order: Order 对象。
        :return: 券商返回的订单ID。
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        请求取消一个活动订单。
        :param order_id: 要取消的订单ID。
        :return: True (如果取消请求被接受) / False (如果失败)。
        """
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> Optional[Order]:
        """
        查询特定订单的当前状态。
        :param order_id: 要查询的订单ID。
        :return: 更新后的 Order 对象，如果未找到则返回 None。
        """
        pass

    @abstractmethod
    def get_all_open_orders(self) -> List[Order]:
        """
        获取所有未结订单。
        :return: Order 对象列表。
        """
        pass

    @abstractmethod
    def get_portfolio_value(self) -> float:
        """
        获取当前投资组合的总价值。
        """
        pass

    @abstractmethod
    def get_cash_balance(self) -> float:
        """
        获取当前可用现金余额。
        """
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> float:
        """
        获取特定资产的持仓数量。
        :param symbol: 资产代码。
        :return: 持仓数量 (正数为多头, 负数为空头, 0为无持仓)。
        """
        pass

    @abstractmethod
    def get_market_data(self, symbol: str, start: datetime, end: datetime) -> List[dict]:
        """
        (可选) 从券商获取历史市场数据。
        """
        pass

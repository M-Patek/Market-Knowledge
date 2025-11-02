"""
测试执行层 (OrderManager, Adapters)
"""
import pytest
from unittest.mock import MagicMock
from datetime import datetime

from execution.order_manager import OrderManager
from execution.interfaces import IBrokerAdapter
from core.schemas.data_schema import Order, Fill, OrderStatus

# FIX (E6): 导入在 E6 修复中添加的 StrategySignal 占位符
from execution.signal_protocol import StrategySignal 

@pytest.fixture
def mock_broker() -> IBrokerAdapter:
    """
    模拟一个 IBrokerAdapter。
    """
    broker = MagicMock(spec=IBrokerAdapter)
    broker.place_order.return_value = "broker_order_123"
    return broker

@pytest.fixture
def order_manager(mock_broker: IBrokerAdapter) -> OrderManager:
    """
    返回一个 OrderManager 实例。
    """
    return OrderManager(broker=mock_broker)

@pytest.fixture
def sample_order() -> Order:
    """
    返回一个示例订单。
    """
    return Order(
        id="client_order_001",
        symbol="MSFT",
        quantity=50.0,
        order_type="MARKET",
        status=OrderStatus.NEW
    )

def test_order_manager_submit(
    order_manager: OrderManager,
    mock_broker: IBrokerAdapter,
    sample_order: Order
):
    """
    测试 OrderManager 是否正确提交订单到券商。
    """
    order_manager.submit_orders([sample_order])
    
    # 验证 broker.place_order 被调用
    mock_broker.place_order.assert_called_once_with(sample_order)
    
    # 验证订单状态在 OrderManager 内部被跟踪
    active_orders = order_manager.get_open_orders()
    assert len(active_orders) == 1
    assert active_orders[0].id == "client_order_001"
    assert active_orders[0].status == OrderStatus.PENDING # OM 将其标记为 PENDING

def test_order_manager_status_update(order_manager: OrderManager):
    """
    测试 OrderManager 如何处理来自 broker 的状态更新回调。
    """
    # 假设 _on_order_status_update 是由 mock_broker 调用的
    
    # 1. 更新为 ACCEPTED
    accepted_order = Order(id="client_order_001", symbol="MSFT", quantity=50, order_type="MARKET", status=OrderStatus.ACCEPTED)
    order_manager._on_order_status_update(accepted_order)
    
    assert order_manager.get_open_orders()[0].status == OrderStatus.ACCEPTED
    
    # 2. 更新为 FILLED
    filled_order = Order(id="client_order_001", symbol="MSFT", quantity=50, order_type="MARKET", status=OrderStatus.FILLED)
    order_manager._on_order_status_update(filled_order)
    
    # 订单完成后应从活动列表中移除
    assert len(order_manager.get_open_orders()) == 0

def test_strategy_signal_import():
    """
    FIX (E6): 简单测试 StrategySignal 是否可以被导入 (因为它现在存在了)。
    """
    signal = StrategySignal(
        symbol="TEST",
        timestamp=datetime.now()
    )
    assert signal.symbol == "TEST"

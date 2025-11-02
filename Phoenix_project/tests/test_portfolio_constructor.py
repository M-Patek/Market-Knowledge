"""
测试 PortfolioConstructor (已更新)
"""
import pytest
from unittest.mock import MagicMock
from datetime import datetime

from cognitive.portfolio_constructor import PortfolioConstructor
from sizing.base import IPositionSizer
from core.schemas.data_schema import Signal, PortfolioState, Position
from core.schemas.fusion_result import FusionResult

# FIX (E10): 重写测试以正确模拟依赖项

@pytest.fixture
def mock_position_sizer() -> IPositionSizer:
    """
    模拟一个 IPositionSizer。
    """
    sizer = MagicMock(spec=IPositionSizer)
    # 模拟计算：返回 100 股
    sizer.calculate_target_quantity.return_value = 100.0
    return sizer

@pytest.fixture
def portfolio_constructor(mock_position_sizer: IPositionSizer) -> PortfolioConstructor:
    """
    返回一个带有模拟 Sizer 的 PortfolioConstructor 实例。
    """
    return PortfolioConstructor(position_sizer=mock_position_sizer)

@pytest.fixture
def bullish_fusion_result() -> FusionResult:
    """
    返回一个看涨的 FusionResult。
    """
    return FusionResult(
        id="fusion1",
        timestamp=datetime(2023, 1, 1, 10, 0, 0),
        agent_decisions=[], # 在 translate 中不重要
        final_decision="STRONG_BUY",
        final_confidence=0.9,
        summary="Test",
        uncertainty_score=0.1,
        metadata={"target_symbol": "AAPL"}
    )

@pytest.fixture
def empty_portfolio_state() -> PortfolioState:
    """
    返回一个空的投资组合状态。
    """
    return PortfolioState(
        timestamp=datetime(2023, 1, 1, 9, 0, 0),
        cash=100000.0,
        total_value=100000.0,
        positions={}
    )

@pytest.fixture
def existing_portfolio_state() -> PortfolioState:
    """
    返回一个已有 AAPL 仓位的投资组合状态。
    """
    return PortfolioState(
        timestamp=datetime(2023, 1, 1, 9, 0, 0),
        cash=85000.0,
        total_value=100000.0,
        positions={
            "AAPL": Position(
                symbol="AAPL",
                quantity=50.0, # 已有 50 股
                average_price=150.0,
                market_value=7500.0,
                unrealized_pnl=0.0
            )
        }
    )

def test_translate_decision_to_signal(portfolio_constructor: PortfolioConstructor, bullish_fusion_result: FusionResult):
    """
    测试从 FusionResult 到 Signal 的转换。
    """
    signal = portfolio_constructor.translate_decision_to_signal(bullish_fusion_result)
    
    assert signal is not None
    assert signal.symbol == "AAPL"
    assert signal.signal_type == "BUY"
    assert signal.strength == 0.9

def test_generate_orders_new_position(
    portfolio_constructor: PortfolioConstructor,
    mock_position_sizer: IPositionSizer,
    bullish_fusion_result: FusionResult,
    empty_portfolio_state: PortfolioState
):
    """
    测试在没有仓位时生成订单。
    """
    # 1. 转换
    signal = portfolio_constructor.translate_decision_to_signal(bullish_fusion_result)
    
    # 2. 生成订单
    orders = portfolio_constructor.generate_orders([signal], empty_portfolio_state)
    
    # 3. 验证
    # 检查 sizer 被正确调用
    mock_position_sizer.calculate_target_quantity.assert_called_with(signal, empty_portfolio_state)
    
    assert len(orders) == 1
    order = orders[0]
    assert order.symbol == "AAPL"
    # 目标 = 100, 当前 = 0 -> 订单 = 100
    assert order.quantity == 100.0

def test_generate_orders_adjust_position(
    portfolio_constructor: PortfolioConstructor,
    mock_position_sizer: IPositionSizer,
    bullish_fusion_result: FusionResult,
    existing_portfolio_state: PortfolioState
):
    """
    测试在已有仓位时调整订单。
    """
    # 1. 转换
    signal = portfolio_constructor.translate_decision_to_signal(bullish_fusion_result)
    
    # 2. 生成订单
    orders = portfolio_constructor.generate_orders([signal], existing_portfolio_state)
    
    # 3. 验证
    # Sizer 仍然返回 100 (这是我们的 mock)
    mock_position_sizer.calculate_target_quantity.assert_called_with(signal, existing_portfolio_state)
    
    assert len(orders) == 1
    order = orders[0]
    assert order.symbol == "AAPL"
    # 目标 = 100, 当前 = 50 -> 订单 = 50
    assert order.quantity == 50.0

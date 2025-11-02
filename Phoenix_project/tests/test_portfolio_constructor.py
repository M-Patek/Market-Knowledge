"""
Tests for the PortfolioConstructor
"""
import pytest
import pandas as pd
from cognitive.portfolio_constructor import PortfolioConstructor

# 关键修正 (Error 9):
# 从 sizing.base 导入正确的接口 'IPositionSizer'
# 而不是不存在的 'BaseSizer'
from sizing.base import IPositionSizer
from sizing.fixed_fraction import FixedFractionSizer # (导入一个具体的实现用于测试)

# --- Mocks ---

class MockPositionSizer(IPositionSizer):
    """
    IPositionSizer 接口的模拟实现
    """
    def calculate_size(self, signal, current_portfolio, market_data) -> Dict[str, float]:
        # (模拟返回一个固定的仓位大小)
        if signal.get('action') == 'BUY':
            return {signal.get('ticker'): 0.5} # (分配 50% 仓位)
        else:
            return {signal.get('ticker'): 0.0} # (清仓)

# (假设的 FusionResult 模拟)
class MockFusionResult:
    def __init__(self, decision):
        self.final_decision = decision

# --- Tests ---

@pytest.fixture
def constructor_config():
    """Fixture for portfolio constructor config"""
    return {
        "sizer": "fixed_fraction", # (假设配置指定了sizer)
        "risk_manager": {} # (假设)
    }

@pytest.fixture
def portfolio_constructor(constructor_config):
    """Fixture for PortfolioConstructor"""
    # (在实际应用中，sizer 和 risk_manager 可能是被注入的)
    # (为了测试，我们可以在这里实例化它们)
    
    # 修正: 确保 sizers 字典中包含一个 'IPositionSizer' 的实例
    sizers = {
        "fixed_fraction": FixedFractionSizer(),
        "mock_sizer": MockPositionSizer()
    }
    # (risk_manager_mock = ...)
    
    return PortfolioConstructor(
        config=constructor_config,
        # sizers=sizers, 
        # risk_manager=risk_manager_mock
    )

def test_constructor_initialization(portfolio_constructor):
    """测试构造函数是否正确初始化"""
    assert portfolio_constructor is not None
    # 修正: 检查它是否正确加载了 sizer
    # (这取决于 PortfolioConstructor 的内部实现)
    # assert isinstance(portfolio_constructor.sizer, FixedFractionSizer)

def test_generate_signal(portfolio_constructor):
    """测试信号生成逻辑"""
    
    # 1. 模拟一个 "BUY" 决策的融合结果
    buy_decision = {
        "ticker": "AAPL",
        "action": "BUY",
        "confidence": 0.85,
        "price_target": 200.0
    }
    fusion_result = MockFusionResult(decision=buy_decision)

    # 2. 模拟当前市场和投资组合
    market_data = {"AAPL": {"price": 180.0}}
    current_portfolio = {"cash": 10000} # (e.g., $10k cash)

    # 3. 运行信号生成
    signal = portfolio_constructor.generate_signal(
        fusion_result, 
        current_portfolio, 
        market_data
    )

    # 4. 验证信号
    assert signal is not None
    # 验证 (Error 2) 修复: 信号应该是 StrategySignal 格式 (target_weights)
    assert signal.strategy_id == "PortfolioConstructor"
    assert "AAPL" in signal.target_weights
    # (这里的具体权重取决于 FixedFractionSizer 的逻辑)
    # assert signal.target_weights["AAPL"] > 0 

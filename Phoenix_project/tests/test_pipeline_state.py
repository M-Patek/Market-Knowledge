"""
测试 PipelineState (已更新)
"""
import pytest
from datetime import datetime

from core.pipeline_state import PipelineState
from core.schemas.data_schema import MarketData, PortfolioState, Position, NewsData

# FIX (E10): 重写测试以匹配新的构造函数和 API

@pytest.fixture
def empty_state():
    """
    返回一个具有默认配置的新 PipelineState。
    """
    # (E5 Fix) 使用新的构造函数
    return PipelineState(initial_state=None, max_history=10)

@pytest.fixture
def sample_portfolio_state():
    """
    返回一个示例投资组合状态。
    """
    return PortfolioState(
        timestamp=datetime(2023, 1, 1, 10, 0, 0),
        cash=100000.0,
        total_value=105000.0,
        positions={
            "AAPL": Position(
                symbol="AAPL",
                quantity=100.0,
                average_price=150.0,
                market_value=15000.0, # (应为 100 * 150)
                unrealized_pnl=0.0
            )
        }
    )

@pytest.fixture
def sample_market_data():
    """
    返回一个 MarketData 示例。
    """
    return MarketData(
        symbol="AAPL",
        timestamp=datetime(2023, 1, 1, 12, 0, 0),
        open=151.0, high=152.0, low=150.0, close=151.5, volume=10000
    )

def test_pipeline_state_initialization(empty_state: PipelineState):
    """
    测试 PipelineState 是否正确初始化。
    """
    assert empty_state.current_time == datetime.min
    assert empty_state.get_latest_portfolio_state() is None
    assert empty_state.max_history == 10

def test_update_time(empty_state: PipelineState):
    """
    测试时间更新。
    """
    new_time = datetime(2023, 1, 1, 12, 0, 0)
    empty_state.update_time(new_time)
    assert empty_state.current_time == new_time

def test_update_portfolio_state(empty_state: PipelineState, sample_portfolio_state: PortfolioState):
    """
    测试投资组合状态的更新和检索。
    """
    empty_state.update_portfolio_state(sample_portfolio_state)
    
    latest_state = empty_state.get_latest_portfolio_state()
    assert latest_state is not None
    assert latest_state.cash == 100000.0
    assert latest_state.positions["AAPL"].quantity == 100.0

def test_update_data_batch_and_history(empty_state: PipelineState, sample_market_data: MarketData):
    """
    测试数据批次更新和历史记录（deque）。
    """
    sample_news = NewsData(
        id="news1",
        source="Reuters",
        timestamp=datetime(2023, 1, 1, 11, 0, 0),
        symbols=["AAPL"],
        content="Test news."
    )
    
    data_batch = {
        "market_data": [sample_market_data],
        "news_data": [sample_news],
        "economic_indicators": []
    }
    
    empty_state.update_data_batch(data_batch)
    
    assert len(empty_state.market_data_history) == 1
    assert len(empty_state.news_history) == 1
    assert empty_state.market_data_history[0].symbol == "AAPL"
    
    # 测试历史记录的 maxlen
    for i in range(15):
        empty_state.update_data_batch(data_batch)
        
    assert len(empty_state.market_data_history) == empty_state.max_history
    assert len(empty_state.market_data_history) == 10

def test_get_latest_market_data(empty_state: PipelineState):
    """
    测试 get_latest_market_data 辅助函数。
    """
    md1 = MarketData(symbol="AAPL", timestamp=datetime(2023, 1, 1), open=1, high=1, low=1, close=1, volume=1)
    md2 = MarketData(symbol="MSFT", timestamp=datetime(2023, 1, 2), open=1, high=1, low=1, close=1, volume=1)
    md3 = MarketData(symbol="AAPL", timestamp=datetime(2023, 1, 3), open=2, high=2, low=2, close=2, volume=2)
    
    batch = {"market_data": [md1, md2, md3], "news_data": [], "economic_indicators": []}
    empty_state.update_data_batch(batch)
    
    latest_aapl = empty_state.get_latest_market_data("AAPL")
    latest_msft = empty_state.get_latest_market_data("MSFT")
    
    assert latest_aapl is not None
    assert latest_aapl.close == 2.0 # 确保拿到的是最新的
    
    assert latest_msft is not None
    assert latest_msft.close == 1.0

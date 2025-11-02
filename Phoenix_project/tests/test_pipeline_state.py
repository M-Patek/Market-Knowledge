import pytest
import pandas as pd
from datetime import timedelta

# 修正：[FIX-ImportError]
# 将所有 `..` 相对导入更改为从项目根目录开始的绝对导入，
# 以匹配 `conftest.py` 设置的 sys.path 约定。
from core.pipeline_state import PipelineState

@pytest.fixture
def state():
    """Returns a fresh PipelineState instance for each test."""
    return PipelineState(max_recent_events=5)

def test_pipeline_state_init(state):
    """Tests the initial default values of the state."""
    assert state.get_current_time() is not None
    assert isinstance(state.get_current_time(), pd.Timestamp)
    
    market_state = state.get_market_state()
    assert market_state["regime"] == "Unknown"
    assert market_state["volatility"] == 0.0
    assert market_state["sentiment"] == 0.0
    
    portfolio_state = state.get_portfolio_state()
    assert portfolio_state["total_value"] == 0.0
    assert portfolio_state["cash"] == 0.0
    assert portfolio_state["positions"] == {}
    
    system_health = state.get_system_health()
    assert system_health["components"] == {"Orchestrator": "OK"}
    assert system_health["last_error"] is None
    
    assert state.get_recent_events() == []

def test_time_update(state):
    """Tests updating and retrieving the current time."""
    new_time = pd.Timestamp.utcnow() - timedelta(days=1)
    state.update_time(new_time)
    assert state.get_current_time() == new_time

def test_market_state_update(state):
    """Tests updating and retrieving the market state."""
    state.update_market_state(regime="Bull", volatility=0.5, sentiment=0.8)
    market_state = state.get_market_state()
    assert market_state["regime"] == "Bull"
    assert market_state["volatility"] == 0.5
    assert market_state["sentiment"] == 0.8

def test_portfolio_state_update(state):
    """Tests updating and retrieving the portfolio state."""
    new_positions = {"AAPL": 10.0, "MSFT": 5.0}
    state.update_portfolio(total_value=1500.0, cash=500.0, positions=new_positions)
    
    portfolio_state = state.get_portfolio_state()
    assert portfolio_state["total_value"] == 1500.0
    assert portfolio_state["cash"] == 500.0
    assert portfolio_state["positions"] == new_positions
    
    # Test that a copy is returned
    new_positions["GOOG"] = 1.0
    assert state.get_portfolio_state()["positions"] == {"AAPL": 10.0, "MSFT": 5.0}

def test_event_queue(state):
    """Tests the FIFO queue functionality for recent events."""
    assert state._recent_events.maxlen == 5
    
    ev1 = {"id": 1}
    ev2 = {"id": 2}
    ev3 = {"id": 3}
    ev4 = {"id": 4}
    ev5 = {"id": 5}
    ev6 = {"id": 6}
    
    state.add_event(ev1)
    state.add_event(ev2)
    state.add_event(ev3)
    
    assert state.get_recent_events() == [ev3, ev2, ev1]
    
    state.add_event(ev4)
    state.add_event(ev5)
    
    assert state.get_recent_events() == [ev5, ev4, ev3, ev2, ev1]
    
    # Add one more, ev1 should be pushed out
    state.add_event(ev6)
    assert state.get_recent_events() == [ev6, ev5, ev4, ev3, ev2]

def test_system_health_update(state):
    """Tests updating component health and errors."""
    state.update_component_health("DataManager", "Warning", "API key missing")
    
    health = state.get_system_health()
    assert health["components"] == {
        "Orchestrator": "OK",
        "DataManager": "Warning"
    }
    assert health["last_error"] == "[DataManager] API key missing"
    
    state.update_component_health("DataManager", "OK")
    health = state.get_system_health()
    assert health["components"]["DataManager"] == "OK"
    # Last error should persist until cleared (or overwritten)
    assert health["last_error"] == "[DataManager] API key missing"
    
    state.update_component_health("Worker", "Error", "Redis connection failed")
    assert state.get_system_health()["last_error"] == "[Worker] Redis connection failed"

def test_full_state_snapshot(state):
    """Tests the combined snapshot method."""
    time = pd.Timestamp.utcnow()
    state.update_time(time)
    state.update_market_state("Bear", 0.9, -0.5)
    
    snapshot = state.get_full_state_snapshot()
    
    assert snapshot["current_time"] == time
    assert snapshot["market_state"]["regime"] == "Bear"
    assert snapshot["portfolio_state"]["cash"] == 0.0
    assert snapshot["system_health"]["components"]["Orchestrator"] == "OK"
    assert snapshot["recent_events_count"] == 0

import pytest
from datetime import datetime

# 修正：[FIX-ImportError]
# 将所有 `..` 相对导入更改为从项目根目录开始的绝对导入，
# 以匹配 `conftest.py` 设置的 sys.path 约定。
from execution.order_manager import OrderManager
from execution.signal_protocol import StrategySignal
from core.pipeline_state import PipelineState

@pytest.fixture
def config():
    """Provides a mock config for the execution layer."""
    return {
        "execution": {
            "execution_model": {
                "impact_coefficient": 0.1,  # Price impact per (order_vol / day_vol)
                "max_volume_share": 0.25,   # Max order size as % of daily vol
                "min_trade_notional": 1000  # Minimum order value
            }
        }
    }

@pytest.fixture
def order_manager(config):
    """Fixture for the OrderManager."""
    return OrderManager(config.get('execution', {}))

@pytest.fixture
def pipeline_state():
    """Fixture for a PipelineState with an initial portfolio."""
    state = PipelineState()
    state.update_portfolio(
        total_value=100000.0,
        cash=50000.0,
        positions={"AAPL": 500} # 500 shares of AAPL
    )
    # Mock market data in state
    state.update_market_data(
        "AAPL", 
        {"close": 100.0, "volume": 1000000} # $100/share, 1M daily volume
    )
    state.update_market_data(
        "MSFT",
        {"close": 200.0, "volume": 500000} # $200/share, 500k daily volume
    )
    return state

def test_order_manager_init(order_manager):
    """Tests OrderManager initialization."""
    assert order_manager is not None
    assert order_manager.max_volume_share == 0.25
    assert order_manager.min_trade_notional == 1000

@pytest.mark.asyncio
async def test_generate_orders_simple(order_manager, pipeline_state):
    """
    Tests generating orders to adjust a simple portfolio.
    - Current: 50% cash, 50% AAPL (500 shares @ $100 = $50k)
    - Target: 80% AAPL, 20% MSFT
    """
    
    # Target weights
    target_weights = {
        "AAPL": 0.80, # Target $80,000 (800 shares). Need to buy 300.
        "MSFT": 0.20, # Target $20,000 (100 shares). Need to buy 100.
        # Implies 0% cash
    }
    
    signal = StrategySignal(
        strategy_id="test_strat",
        timestamp=datetime.utcnow(),
        target_weights=target_weights
    )

    orders = await order_manager.generate_orders_from_signal(signal, pipeline_state)
    
    assert orders is not None
    assert len(orders) == 2
    
    orders_by_symbol = {o.symbol: o for o in orders}
    
    # Check AAPL order
    assert "AAPL" in orders_by_symbol
    aapl_order = orders_by_symbol["AAPL"]
    assert aapl_order.action == "BUY"
    assert aapl_order.quantity == 300 # 800 (target) - 500 (current)
    assert aapl_order.order_type == "LIMIT"
    # Price impact = 0.1 * (300 / 1000000) * 100 = 0.003
    # Limit price for BUY is higher (worse)
    assert aapl_order.limit_price == pytest.approx(100.0 * (1 + 0.1 * (300/1000000))) 
    
    # Check MSFT order
    assert "MSFT" in orders_by_symbol
    msft_order = orders_by_symbol["MSFT"]
    assert msft_order.action == "BUY"
    assert msft_order.quantity == 100 # $20,000 / $200
    assert msft_order.order_type == "LIMIT"
    assert msft_order.limit_price == pytest.approx(200.0 * (1 + 0.1 * (100/500000)))

@pytest.mark.asyncio
async def test_generate_orders_liquidate(order_manager, pipeline_state):
    """Tests liquidating a position."""
    
    # Target: 100% Cash
    target_weights = {} # Empty means all cash
    
    signal = StrategySignal(
        strategy_id="test_strat",
        timestamp=datetime.utcnow(),
        target_weights=target_weights
    )

    orders = await order_manager.generate_orders_from_signal(signal, pipeline_state)
    
    assert len(orders) == 1
    
    aapl_order = orders[0]
    assert aapl_order.symbol == "AAPL"
    assert aapl_order.action == "SELL"
    assert aapl_order.quantity == 500 # Sell all 500 shares
    assert aapl_order.order_type == "LIMIT"
    # Price impact = 0.1 * (500 / 1000000) * 100 = 0.005
    # Limit price for SELL is lower (worse)
    assert aapl_order.limit_price == pytest.approx(100.0 * (1 - 0.1 * (500/1000000)))

@pytest.mark.asyncio
async def test_order_constraints_volume_limit(order_manager, pipeline_state):
    """Tests that the max_volume_share constraint is enforced."""
    
    # Target 100% MSFT ($100,000)
    # This would be 500 shares.
    # Daily volume is 500,000.
    # Max volume share is 0.25, so max order is 0.25 * 500,000 = 125,000 shares.
    # Order of 500 is fine.
    
    # Let's override the config for this test
    order_manager.max_volume_share = 0.0001 # Max 0.01% of 500k vol = 50 shares
    
    target_weights = {"MSFT": 1.0}
    signal = StrategySignal("test", datetime.utcnow(), target_weights)
    
    orders = await order_manager.generate_orders_from_signal(signal, pipeline_state)
    
    orders_by_symbol = {o.symbol: o for o in orders}

    # Check MSFT order
    assert "MSFT" in orders_by_symbol
    msft_order = orders_by_symbol["MSFT"]
    assert msft_order.action == "BUY"
    # Target was 500 shares, but capped at 50 (0.0001 * 500k)
    assert msft_order.quantity == 50 

@pytest.mark.asyncio
async def test_order_constraints_min_notional(order_manager, pipeline_state):
    """Tests that the min_trade_notional constraint is enforced."""
    
    # Target: 0.1% MSFT ($100)
    # This would be 0.5 shares ($100 / $200).
    # This is below the $1000 min_trade_notional.
    
    target_weights = {
        "AAPL": 0.50, # No change
        "MSFT": 0.001 # $100 target
    }
    signal = StrategySignal("test", datetime.utcnow(), target_weights)
    
    orders = await order_manager.generate_orders_from_signal(signal, pipeline_state)
    
    # The AAPL order (500 -> 500) should be 0.
    # The MSFT order (0 -> 0.5) should be filtered out.
    assert len(orders) == 0

@pytest.mark.asyncio
async def test_order_constraints_no_data(order_manager, pipeline_state):
    """Tests that no order is generated for a symbol with no market data."""
    
    # Target: 50% GOOG, but GOOG data is not in pipeline_state
    target_weights = {"GOOG": 0.5}
    signal = StrategySignal("test", datetime.utcnow(), target_weights)
    
    orders = await order_manager.generate_orders_from_signal(signal, pipeline_state)
    
    # AAPL order (to sell) and GOOG order (no data)
    orders_by_symbol = {o.symbol: o for o in orders}
    
    assert "AAPL" in orders_by_symbol # Sell order for AAPL
    assert "GOOG" not in orders_by_symbol # No order for GOOG

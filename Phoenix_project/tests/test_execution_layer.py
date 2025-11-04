# tests/test_execution_layer.py
import pytest
import time
from unittest.mock import MagicMock, call, patch

# --- [修复] ---
# 修复：将 'execution.order_manager' 转换为 'Phoenix_project.execution.order_manager'
from Phoenix_project.execution.order_manager import OrderManager
# 修复：将 'execution.adapters' 转换为 'Phoenix_project.execution.adapters'
from Phoenix_project.execution.adapters import SimulatedBrokerAdapter
# 修复：将 'execution.trade_lifecycle_manager' 转换为 'Phoenix_project.execution.trade_lifecycle_manager'
from Phoenix_project.execution.trade_lifecycle_manager import TradeLifecycleManager
# 修复：将 'core.schemas.data_schema' 转换为 'Phoenix_project.core.schemas.data_schema'
from Phoenix_project.core.schemas.data_schema import Order, Fill, OrderStatus, Position, PortfolioState
# 修复：将 'monitor.logging' 转换为 'Phoenix_project.monitor.logging'
from Phoenix_project.monitor.logging import get_logger
# --- [修复结束] ---


# We need a logger for the components
logger = get_logger("test_execution")

@pytest.fixture
def simulated_broker():
    """Fixture for a clean SimulatedBrokerAdapter."""
    return SimulatedBrokerAdapter(initial_cash=100000.0)

@pytest.fixture
def order_manager(simulated_broker):
    """Fixture for an OrderManager connected to the sim broker."""
    # We patch the logger inside OrderManager to avoid noise
    with patch("Phoenix_project.execution.order_manager.logger", MagicMock()):
        om = OrderManager(broker=simulated_broker)
    return om

@pytest.fixture
def trade_manager():
    """Fixture for a TradeLifecycleManager."""
    with patch("Phoenix_project.execution.trade_lifecycle_manager.logger", MagicMock()):
        tm = TradeLifecycleManager(initial_cash=100000.0)
    return tm

# --- Test OrderManager ---

def test_order_manager_submit_order(order_manager, simulated_broker):
    """
    Tests if OrderManager correctly submits an order to the broker
    and tracks it as active.
    """
    order = Order(
        id="order_001",
        symbol="AAPL",
        quantity=10,
        order_type="MARKET",
        timestamp=time.time()
    )
    
    # Spy on the broker's place_order method
    simulated_broker.place_order = MagicMock(return_value="broker_id_001")
    
    order_manager.submit_orders([order])
    
    # 1. Verify broker was called
    simulated_broker.place_order.assert_called_once_with(order)
    
    # 2. Verify order is in active list
    active_orders = order_manager.get_open_orders()
    assert len(active_orders) == 1
    assert active_orders[0].id == "order_001"
    assert active_orders[0].status == OrderStatus.PENDING # OM marks as PENDING

def test_order_manager_handles_callbacks(order_manager, simulated_broker):
    """
    Tests if OrderManager correctly processes Fill and Status callbacks
    from the broker and updates its internal state.
    """
    order = Order(
        id="order_002",
        symbol="MSFT",
        quantity=5,
        order_type="MARKET",
        timestamp=time.time()
    )
    
    # 1. Submit the order first
    order_manager.submit_orders([order])
    assert len(order_manager.get_open_orders()) == 1

    # 2. Simulate Broker ACCEPTED callback
    order.status = OrderStatus.ACCEPTED
    simulated_broker.order_status_callback(order) # Manually trigger callback
    
    assert order_manager.get_open_orders()[0].status == OrderStatus.ACCEPTED
    
    # 3. Simulate Broker FILLED callback
    fill = Fill(
        id="fill_001",
        order_id="order_002",
        symbol="MSFT",
        quantity=5,
        price=100.0,
        commission=0.0,
        timestamp=time.time()
    )
    order.status = OrderStatus.FILLED
    
    # Broker sends Fill first, then final OrderStatus
    simulated_broker.fill_callback(fill)
    simulated_broker.order_status_callback(order)
    
    # 4. Verify order is removed from active list
    assert len(order_manager.get_open_orders()) == 0

# --- Test TradeLifecycleManager ---

def test_trade_manager_buy_fill(trade_manager):
    """Tests processing a simple buy Fill."""
    fill = Fill(
        id="f_001", order_id="o_001", symbol="AAPL",
        quantity=10, price=150.0, commission=1.0, timestamp=time.time()
    )
    
    trade_manager.on_fill(fill)
    
    # 1. Check cash
    # 100000 - (10 * 150) - 1.0 = 100000 - 1500 - 1 = 98499.0
    assert trade_manager.cash == 98499.0
    
    # 2. Check position
    assert "AAPL" in trade_manager.positions
    pos = trade_manager.positions["AAPL"]
    assert pos.symbol == "AAPL"
    assert pos.quantity == 10
    assert pos.average_price == 150.0

def test_trade_manager_add_to_position(trade_manager):
    """Tests averaging up a position."""
    fill_1 = Fill(
        id="f_001", order_id="o_001", symbol="AAPL",
        quantity=10, price=150.0, commission=1.0, timestamp=time.time()
    )
    fill_2 = Fill(
        id="f_002", order_id="o_002", symbol="AAPL",
        quantity=5, price=160.0, commission=0.5, timestamp=time.time()
    )
    
    trade_manager.on_fill(fill_1)
    trade_manager.on_fill(fill_2)

    # 1. Check cash
    # Start: 100000
    # Fill 1: -1501.0 (Cash = 98499.0)
    # Fill 2: -(5 * 160) - 0.5 = -800.5 (Cash = 97698.5)
    assert trade_manager.cash == 97698.5
    
    # 2. Check position
    assert "AAPL" in trade_manager.positions
    pos = trade_manager.positions["AAPL"]
    assert pos.quantity == 15 # 10 + 5
    
    # Avg Price: ((10 * 150) + (5 * 160)) / 15 = (1500 + 800) / 15 = 2300 / 15
    expected_avg_price = 2300.0 / 15.0
    assert pos.average_price == pytest.approx(expected_avg_price)

def test_trade_manager_close_position(trade_manager):
    """Tests closing a position and calculating realized PnL."""
    fill_1 = Fill(
        id="f_001", order_id="o_001", symbol="AAPL",
        quantity=10, price=150.0, commission=1.0, timestamp=time.time()
    )
    # Sell fill
    fill_2 = Fill(
        id="f_002", order_id="o_002", symbol="AAPL",
        quantity=-10, price=170.0, commission=1.0, timestamp=time.time()
    )
    
    trade_manager.on_fill(fill_1) # Buy 10 @ 150 (Cash: 98499.0)
    trade_manager.on_fill(fill_2) # Sell 10 @ 170

    # 1. Check cash
    # Start: 98499.0
    # Fill 2: +(-10 * 170) - 1.0 = +1700 - 1.0 = +1699.0
    # Cash = 98499.0 + 1699.0 = 100198.0
    assert trade_manager.cash == 100198.0
    
    # 2. Check position (should be closed)
    assert "AAPL" not in trade_manager.positions
    
    # 3. Check Realized PnL
    # PnL = (Sell Price - Buy Price) * Quantity
    # PnL = (170.0 - 150.0) * 10 = 20.0 * 10 = 200.0
    # Note: Commissions are handled separately in cash, not in PnL calc here.
    assert trade_manager.realized_pnl == 200.0

def test_trade_manager_get_portfolio_state(trade_manager):
    """Tests the calculation of unrealized PnL and total value."""
    fill_1 = Fill(
        id="f_001", order_id="o_001", symbol="AAPL",
        quantity=10, price=150.0, commission=1.0, timestamp=time.time()
    )
    trade_manager.on_fill(fill_1) # Cash: 98499.0, Pos: 10 @ 150
    
    # 1. Define current market prices
    market_data = {"AAPL": 160.0}
    
    # 2. Get state
    state = trade_manager.get_current_portfolio_state(market_data)
    
    assert state.cash == 98499.0
    
    # 3. Check position details in state
    assert "AAPL" in state.positions
    pos_state = state.positions["AAPL"]
    
    # Unrealized PnL = (Current Price - Avg Price) * Qty
    # (160.0 - 150.0) * 10 = 100.0
    assert pos_state.unrealized_pnl == 100.0
    
    # Market Value = Current Price * Qty
    # 160.0 * 10 = 1600.0
    assert pos_state.market_value == 1600.0
    
    # 4. Check Total Value
    # Total Value = Cash + Market Value
    # 98499.0 + 1600.0 = 100099.0
    assert state.total_value == 100099.0

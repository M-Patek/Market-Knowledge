# tests/test_execution_layer.py
import pytest
import time
from unittest.mock import MagicMock, call, patch, ANY
from datetime import datetime # [任务 2 修复] 导入 datetime

# --- [修复] ---
# 修复：将 'execution.order_manager' 转换为 'Phoenix_project.execution.order_manager'
from Phoenix_project.execution.order_manager import OrderManager
# 修复：将 'execution.adapters' 转换为 'Phoenix_project.execution.adapters'
from Phoenix_project.execution.adapters import SimulatedBrokerAdapter
# 修复：将 'execution.trade_lifecycle_manager' 转换为 'Phoenix_project.execution.trade_lifecycle_manager'
from Phoenix_project.execution.trade_lifecycle_manager import TradeLifecycleManager
# 修复：将 'core.schemas.data_schema' 转换为 'Phoenix_project.core.schemas.data_schema'
from Phoenix_project.core.schemas.data_schema import Order, Fill, OrderStatus, Position, PortfolioState, TargetPortfolio, TargetPosition
# 修复：将 'monitor.logging' 转换为 'Phoenix_project.monitor.logging'
from Phoenix_project.monitor.logging import get_logger
# --- [修复结束] ---


# We need a logger for the components
logger = get_logger("test_execution")

@pytest.fixture
def trade_manager():
    """[任务 2 修复] Fixture for a TradeLifecycleManager."""
    with patch("Phoenix_project.execution.trade_lifecycle_manager.logger", MagicMock()):
        tm = TradeLifecycleManager(initial_cash=100000.0)
    return tm

@pytest.fixture
def simulated_broker():
    """[任务 2 修复] Fixture for a clean SimulatedBrokerAdapter."""
    # 构造函数现在为空
    broker = SimulatedBrokerAdapter()
    # 模拟回调
    broker.fill_callback = MagicMock()
    broker.order_status_callback = MagicMock()
    return broker

@pytest.fixture
def order_manager(simulated_broker, trade_manager):
    """[任务 2 修复] Fixture for an OrderManager."""
    with patch("Phoenix_project.execution.order_manager.logger", MagicMock()):
        # OrderManager 现在需要 broker 和 trade_manager
        om = OrderManager(
            broker=simulated_broker, 
            trade_lifecycle_manager=trade_manager
        )
    return om

# --- [任务 2 修复] 新增测试: 测试新的核心逻辑 ---
def test_order_manager_process_target_portfolio_buy(order_manager, simulated_broker, trade_manager):
    """
    Tests the new `process_target_portfolio` logic for a BUY order.
    """
    # 1. 初始状态 (全现金)
    current_prices = {"AAPL": 150.0}
    current_portfolio = trade_manager.get_current_portfolio_state(current_prices)
    assert current_portfolio.total_value == 100000.0
    
    # 2. 目标状态 (买入 10% AAPL)
    target_portfolio = TargetPortfolio(
        positions=[
            TargetPosition(symbol="AAPL", target_weight=0.10, reasoning="Test buy")
        ]
    )
    
    # 3. 模拟 SimBroker.place_order
    # (SimBroker 会立即执行并触发回调, 我们需要模拟它)
    
    # 我们 patch `place_order` 来 *手动* 触发回调, 模拟 SimBroker 的行为
    def mock_place_order(order, price):
        fill = Fill(
            id=f"fill_{order.id}",
            order_id=order.id,
            symbol=order.symbol,
            timestamp=datetime.utcnow(),
            quantity=order.quantity,
            price=price, # 假设无滑点
            commission=0.0
        )
        order.status = OrderStatus.FILLED
        
        # 模拟 SimBroker 触发回调 (这会命中 OrderManager 的 _on_fill)
        simulated_broker.order_status_callback(order) # 状态更新
        simulated_broker.fill_callback(fill)         # 成交
        return f"broker_{order.id}"

    simulated_broker.place_order = MagicMock(side_effect=mock_place_order)

    # 4. 执行
    order_manager.process_target_portfolio(
        current_portfolio, 
        target_portfolio, 
        current_prices
    )
    
    # 5. 验证是否调用了 place_order
    # 目标 $10,000 (10% of 100k) @ $150/share = 66.666... shares
    expected_qty = 10000.0 / 150.0
    
    simulated_broker.place_order.assert_called_once()
    # 验证第一个参数 (order)
    call_args = simulated_broker.place_order.call_args[0]
    assert isinstance(call_args[0], Order)
    assert call_args[0].symbol == "AAPL"
    assert call_args[0].quantity == pytest.approx(expected_qty)
    # 验证第二个参数 (price)
    assert call_args[1] == 150.0

    # 6. 验证 TradeLifecycleManager (因为回调被触发)
    assert trade_manager.cash == pytest.approx(100000.0 - 10000.0)
    assert "AAPL" in trade_manager.positions
    assert trade_manager.positions["AAPL"].quantity == pytest.approx(expected_qty)

def test_order_manager_process_target_portfolio_sell(order_manager, simulated_broker, trade_manager):
    """
    Tests the `process_target_portfolio` logic for a SELL order (reducing position).
    """
    # 1. 初始状态 (持有 $15,000 AAPL, 10% 仓位 @ $150k 总价值)
    initial_fill = Fill(id="f_000", order_id="o_000", symbol="AAPL", quantity=100, price=150.0, commission=0.0, timestamp=time.time())
    trade_manager.on_fill(initial_fill) # Cash: -15k, Pos: 100
    trade_manager.cash += 150000 - 100000 # 重置总价值为 $150k (Cash $135k)
    
    current_prices = {"AAPL": 150.0}
    current_portfolio = trade_manager.get_current_portfolio_state(current_prices)
    assert current_portfolio.total_value == 150000.0
    assert current_portfolio.positions["AAPL"].quantity == 100
    
    # 2. 目标状态 (减少到 5% 仓位)
    target_portfolio = TargetPortfolio(
        positions=[
            TargetPosition(symbol="AAPL", target_weight=0.05, reasoning="Test sell")
        ]
    )
    
    # 3. 模拟 SimBroker (与上一个测试相同)
    def mock_place_order(order, price):
        fill = Fill(id=f"fill_{order.id}", order_id=order.id, symbol=order.symbol, timestamp=datetime.utcnow(), quantity=order.quantity, price=price, commission=0.0)
        order.status = OrderStatus.FILLED
        simulated_broker.order_status_callback(order)
        simulated_broker.fill_callback(fill)
        return f"broker_{order.id}"
    simulated_broker.place_order = MagicMock(side_effect=mock_place_order)

    # 4. 执行
    order_manager.process_target_portfolio(
        current_portfolio, 
        target_portfolio, 
        current_prices
    )
    
    # 5. 验证
    # 目标 $7,500 (5% of 150k) @ $150/share = 50 shares
    # 当前 100 shares
    # 订单 = 50 - 100 = -50 shares (卖出)
    expected_qty = -50.0
    
    simulated_broker.place_order.assert_called_once()
    call_args = simulated_broker.place_order.call_args[0]
    assert call_args[0].symbol == "AAPL"
    assert call_args[0].quantity == pytest.approx(expected_qty)

    # 6. 验证 TLM
    # 最终持仓 50 股
    assert "AAPL" in trade_manager.positions
    assert trade_manager.positions["AAPL"].quantity == pytest.approx(50.0)
    # 初始 Cash 135k。卖出 50 * 150 = $7,500
    assert trade_manager.cash == pytest.approx(135000.0 + 7500.0)
    # PnL (150 - 150) * 50 = 0
    assert trade_manager.realized_pnl == 0.0

# --- [任务 2 修复] 保留 TLM 测试, 仅修复 fixture ---

def test_trade_manager_buy_fill(trade_manager):
    """Tests processing a simple buy Fill. (Fixture fixed)"""
    fill = Fill(
        id="f_001", order_id="o_001", symbol="AAPL",
        quantity=10, price=150.0, commission=1.0, timestamp=time.time()
    )
    
    trade_manager.on_fill(fill)
    
    assert trade_manager.cash == 98499.0
    assert "AAPL" in trade_manager.positions
    pos = trade_manager.positions["AAPL"]
    assert pos.quantity == 10
    assert pos.average_price == 150.0

def test_trade_manager_add_to_position(trade_manager):
    """Tests averaging up a position. (Fixture fixed)"""
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

    assert trade_manager.cash == 97698.5
    assert "AAPL" in trade_manager.positions
    pos = trade_manager.positions["AAPL"]
    assert pos.quantity == 15
    expected_avg_price = (1500.0 + 800.0) / 15.0
    assert pos.average_price == pytest.approx(expected_avg_price)

def test_trade_manager_close_position(trade_manager):
    """Tests closing a position and calculating realized PnL. (Fixture fixed)"""
    fill_1 = Fill(
        id="f_001", order_id="o_001", symbol="AAPL",
        quantity=10, price=150.0, commission=1.0, timestamp=time.time()
    )
    fill_2 = Fill(
        id="f_002", order_id="o_002", symbol="AAPL",
        quantity=-10, price=170.0, commission=1.0, timestamp=time.time()
    )
    
    trade_manager.on_fill(fill_1)
    trade_manager.on_fill(fill_2)

    assert trade_manager.cash == 100198.0
    assert "AAPL" not in trade_manager.positions
    assert trade_manager.realized_pnl == 200.0

def test_trade_manager_get_portfolio_state(trade_manager):
    """Tests the calculation of unrealized PnL and total value. (Fixture fixed)"""
    fill_1 = Fill(
        id="f_001", order_id="o_001", symbol="AAPL",
        quantity=10, price=150.0, commission=1.0, timestamp=time.time()
    )
    trade_manager.on_fill(fill_1)
    
    market_data = {"AAPL": 160.0}
    state = trade_manager.get_current_portfolio_state(market_data)
    
    assert state.cash == 98499.0
    assert "AAPL" in state.positions
    pos_state = state.positions["AAPL"]
    assert pos_state.unrealized_pnl == 100.0
    assert pos_state.market_value == 1600.0
    assert state.total_value == 100099.0

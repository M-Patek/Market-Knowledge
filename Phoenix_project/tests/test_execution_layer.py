# tests/test_execution_layer.py

import pytest
from unittest.mock import MagicMock, Mock
import backtrader as bt
from execution.interfaces import IBrokerAdapter, Order
from execution.order_manager import OrderManager
from execution.adapters import BacktraderBrokerAdapter

# --- Mocks and Fixtures ---

@pytest.fixture
def mock_broker_adapter() -> Mock:
    """Mocks the IBrokerAdapter interface."""
    adapter = Mock(spec=IBrokerAdapter)
    adapter.get_portfolio_value.return_value = 100000.0
    # The place_order mock simply returns the order it received.
    adapter.place_order.side_effect = lambda strategy, order: order
    return adapter

@pytest.fixture
def mock_strategy() -> Mock:
    """Mocks a backtrader.Strategy object with necessary components."""
    strategy = Mock(spec=['getdatabyname', 'getposition', 'buy', 'sell'])
    strategy.datas = []
    
    # Mock the data feed
    mock_data = Mock()
    mock_data.close = [100.0]
    mock_data.volume = [10000.0]
    mock_data.high = [102.0]
    mock_data.low = [98.0]
    mock_data._name = "TEST_TICKER"
    
    # Mock the position object
    mock_position = Mock()
    mock_position.size = 100 # Start with an existing position
    
    strategy.getdatabyname.return_value = mock_data
    strategy.getposition.return_value = mock_position
    
    return strategy

# --- OrderManager Tests ---

def test_order_manager_calculates_buy_order(mock_broker_adapter, mock_strategy):
    """Test that a simple buy order is calculated correctly."""
    order_manager = OrderManager(broker_adapter=mock_broker_adapter)
    
    # Target: 20% allocation ($20,000). Current: 100 shares * $100 = $10,000. Delta: +$10,000
    target_portfolio = [{"ticker": "TEST_TICKER", "capital_allocation_pct": 0.20}]
    
    order_manager.rebalance(mock_strategy, target_portfolio)
    
    mock_broker_adapter.place_order.assert_called_once()
    placed_order: Order = mock_broker_adapter.place_order.call_args[0][1]
    
    assert placed_order.side == 'BUY'
    assert placed_order.ticker == "TEST_TICKER"
    assert pytest.approx(placed_order.size) == 100.0 # $10,000 / $100/share

def test_order_manager_calculates_sell_order(mock_broker_adapter, mock_strategy):
    """Test that a simple sell order is calculated correctly."""
    order_manager = OrderManager(broker_adapter=mock_broker_adapter)

    # Target: 5% allocation ($5,000). Current: $10,000. Delta: -$5,000
    target_portfolio = [{"ticker": "TEST_TICKER", "capital_allocation_pct": 0.05}]

    order_manager.rebalance(mock_strategy, target_portfolio)

    mock_broker_adapter.place_order.assert_called_once()
    placed_order: Order = mock_broker_adapter.place_order.call_args[0][1]

    assert placed_order.side == 'SELL'
    assert placed_order.ticker == "TEST_TICKER"
    assert pytest.approx(placed_order.size) == 50.0 # $5,000 / $100/share

def test_order_manager_applies_liquidity_constraint(mock_broker_adapter, mock_strategy):
    """Tests that the order size is capped by the max_volume_share."""
    # Max share is 2.5% of 10,000 volume = 250 shares
    order_manager = OrderManager(broker_adapter=mock_broker_adapter, max_volume_share=0.025)
    
    # Target: 50% allocation ($50,000). Current: $10,000. Delta: +$40,000 (400 shares)
    # This exceeds the liquidity limit.
    target_portfolio = [{"ticker": "TEST_TICKER", "capital_allocation_pct": 0.50}]

    order_manager.rebalance(mock_strategy, target_portfolio)

    mock_broker_adapter.place_order.assert_called_once()
    placed_order: Order = mock_broker_adapter.place_order.call_args[0][1]

    assert placed_order.side == 'BUY'
    assert pytest.approx(placed_order.size) == 250.0 # Capped at 2.5% of volume

def test_order_manager_respects_min_notional(mock_broker_adapter, mock_strategy):
    """Tests that no order is placed if the value delta is below the minimum."""
    order_manager = OrderManager(broker_adapter=mock_broker_adapter, min_trade_notional=500.0)
    
    # Target: 10.1% ($10,100). Current: $10,000. Delta: +$100.
    # This is below the min_trade_notional.
    target_portfolio = [{"ticker": "TEST_TICKER", "capital_allocation_pct": 0.101}]

    order_manager.rebalance(mock_strategy, target_portfolio)
    
    mock_broker_adapter.place_order.assert_not_called()

# --- BacktraderBrokerAdapter Tests ---

def test_adapter_places_buy_order(mock_strategy):
    """Tests if the adapter correctly calls strategy.buy."""
    adapter = BacktraderBrokerAdapter(broker=MagicMock())
    order = Order(ticker="TEST_TICKER", side="BUY", size=50.0, limit_price=101.0)
    
    result_order = adapter.place_order(mock_strategy, order)
    
    mock_strategy.buy.assert_called_once_with(
        data=mock_strategy.getdatabyname.return_value,
        size=50.0,
        price=101.0,
        exectype=bt.Order.Limit
    )
    assert result_order.status == 'SUBMITTED'

def test_adapter_places_sell_order(mock_strategy):
    """Tests if the adapter correctly calls strategy.sell."""
    adapter = BacktraderBrokerAdapter(broker=MagicMock())
    order = Order(ticker="TEST_TICKER", side="SELL", size=50.0, limit_price=99.0)

    result_order = adapter.place_order(mock_strategy, order)

    mock_strategy.sell.assert_called_once_with(
        data=mock_strategy.getdatabyname.return_value,
        size=50.0,
        price=99.0,
        exectype=bt.Order.Limit
    )
    assert result_order.status == 'SUBMITTED'

def test_adapter_rejects_unfillable_buy_order(mock_strategy):
    """Tests that a buy order with a limit price below the bar's low is rejected."""
    adapter = BacktraderBrokerAdapter(broker=MagicMock())
    # Limit price of $97 is below the bar's low of $98
    order = Order(ticker="TEST_TICKER", side="BUY", size=50.0, limit_price=97.0)
    
    result_order = adapter.place_order(mock_strategy, order)
    
    mock_strategy.buy.assert_not_called()
    assert result_order.status == 'REJECTED'

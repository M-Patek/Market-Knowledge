import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from ..strategy_handler import StrategyDataHandler
from ..core.schemas.data_schema import TickerData, MarketEvent

# Mock StrategyConfig class or dictionary
@pytest.fixture
def mock_config():
    """Provides a mock configuration object."""
    config = {
        'strategy_name': 'TestStrategy',
        'asset_universe': ['AAPL', 'MSFT'],
        'data_lookback_days': 60,
        'resample_frequency': 'D' # Daily
    }
    return config

@pytest.fixture
def mock_data_manager(mock_config):
    """Mocks the DataManager."""
    dm = MagicMock()
    
    # Setup mock historical data
    dates = pd.date_range(end=pd.Timestamp.now(), periods=mock_config['data_lookback_days'] * 2)
    aapl_data = pd.DataFrame({'close': range(len(dates))}, index=dates)
    msft_data = pd.DataFrame({'close': range(len(dates), len(dates) * 2)}, index=dates)
    
    dm.get_historical_data.side_effect = lambda symbol, start, end: \
        aapl_data.loc[start:end] if symbol == 'AAPL' else \
        (msft_data.loc[start:end] if symbol == 'MSFT' else pd.DataFrame())
        
    return dm

@pytest.fixture
def handler(mock_config, mock_data_manager):
    """Initializes the StrategyDataHandler with mocks."""
    return StrategyDataHandler(mock_config, mock_data_manager)

# --- Test Cases ---

def test_initialization(handler, mock_config):
    """Tests if the handler initializes correctly."""
    assert handler.strategy_name == 'TestStrategy'
    assert set(handler.asset_universe) == {'AAPL', 'MSFT'}
    assert handler.data_lookback == pd.Timedelta(days=mock_config['data_lookback_days'])
    assert 'AAPL' not in handler.market_data # Should be empty initially
    assert 'MSFT' not in handler.market_data

def test_preload_historical_data(handler, mock_data_manager, mock_config):
    """Tests the preloading of historical data for the asset universe."""
    end_date = pd.Timestamp('2023-01-31')
    handler.preload_historical_data(end_date)
    
    # Check if DataManager was called correctly
    assert mock_data_manager.get_historical_data.call_count == len(mock_config['asset_universe'])
    
    # Check if data is loaded into the handler's state
    assert 'AAPL' in handler.market_data
    assert 'MSFT' in handler.market_data
    assert len(handler.market_data['AAPL']) == mock_config['data_lookback_days']
    assert len(handler.market_data['MSFT']) == mock_config['data_lookback_days']
    
    # Check if data is correctly sliced
    assert handler.market_data['AAPL'].index.max() <= end_date
    assert handler.market_data['AAPL'].index.min() == end_date - pd.Timedelta(days=mock_config['data_lookback_days'] - 1)


def test_update_market_data_new_symbol(handler):
    """Tests adding a new TickerData point for a symbol not in the lookback."""
    ticker = TickerData(
        symbol='GOOG',
        timestamp=pd.Timestamp('2023-02-01'),
        open=100, high=101, low=99, close=100, volume=1000,
        source='test'
    )
    handler.update_market_data(ticker)
    
    # GOOG is not in the asset universe, so it should be ignored
    assert 'GOOG' not in handler.market_data

def test_update_market_data_existing_symbol(handler):
    """Tests updating an existing symbol with a new TickerData point."""
    # Preload AAPL
    end_date = pd.Timestamp('2023-01-31')
    handler.preload_historical_data(end_date)
    initial_length = len(handler.market_data['AAPL'])
    
    # Add new data point
    new_ticker = TickerData(
        symbol='AAPL',
        timestamp=pd.Timestamp('2023-02-01'),
        open=150, high=151, low=149, close=150, volume=5000,
        source='test'
    )
    handler.update_market_data(new_ticker)
    
    # Length should remain the same (due to lookback window)
    assert len(handler.market_data['AAPL']) == initial_length
    # The new data point should be the last one
    assert handler.market_data['AAPL'].index[-1] == new_ticker.timestamp
    assert handler.market_data['AAPL']['close'].iloc[-1] == new_ticker.close
    # The oldest data point should be gone
    assert handler.market_data['AAPL'].index[0] > end_date - handler.data_lookback

def test_update_market_data_duplicate(handler):
    """Tests that duplicate (same timestamp) data is handled (e.g., ignored or overwritten)."""
    end_date = pd.Timestamp('2023-01-31')
    handler.preload_historical_data(end_date)
    
    # Create a data point with the same timestamp as the last one
    last_timestamp = handler.market_data['AAPL'].index[-1]
    last_close = handler.market_data['AAPL']['close'].iloc[-1]
    
    new_ticker = TickerData(
        symbol='AAPL',
        timestamp=last_timestamp,
        open=999, high=999, low=999, close=999, volume=999, # New data
        source='test'
    )
    
    handler.update_market_data(new_ticker)
    
    # Check if the data was overwritten
    assert handler.market_data['AAPL']['close'].iloc[-1] == 999
    # Ensure no new row was added
    assert len(handler.market_data['AAPL']) == handler.data_lookback.days

def test_update_market_event(handler):
    """Tests the handling of new market events."""
    event = MarketEvent(
        event_id='evt123',
        timestamp=pd.Timestamp('2023-02-01'),
        source='test_source',
        headline='Test Event',
        symbols=['AAPL']
    )
    handler.update_market_event(event)
    
    assert len(handler.market_events) == 1
    assert handler.market_events[0].event_id == 'evt123'
    
    # Test that events for non-universe symbols are filtered
    event_other = MarketEvent(
        event_id='evt456',
        timestamp=pd.Timestamp('2023-02-02'),
        source='test_source',
        headline='Other Event',
        symbols=['GOOG'] # Not in universe
    )
    handler.update_market_event(event_other)
    
    assert len(handler.market_events) == 1 # Should not have been added

def test_get_strategy_state(handler):
    """Tests the snapshotting of the current state."""
    preload_date = pd.Timestamp('2023-01-31')
    handler.preload_historical_data(preload_date)
    
    event = MarketEvent(event_id='evt1', timestamp=preload_date, source='test', headline='H1', symbols=['AAPL'])
    handler.update_market_event(event)
    
    current_time = pd.Timestamp('2023-02-01')
    state = handler.get_strategy_state(current_time)
    
    assert state.timestamp == current_time
    assert state.strategy_name == 'TestStrategy'
    assert 'AAPL' in state.market_data
    assert 'MSFT' in state.market_data
    assert len(state.market_data['AAPL']) == handler.data_lookback.days
    assert len(state.recent_events) == 1
    assert state.recent_events[0].event_id == 'evt1'

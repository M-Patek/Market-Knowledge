# tests/test_data_manager.py
import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
import pandas as pd

# --- [修复] ---
# 修复：将 'data_manager' 转换为 'Phoenix_project.data_manager'
from Phoenix_project.data_manager import DataManager
# 修复：将 'core.schemas.data_schema' 转换为 'Phoenix_project.core.schemas.data_schema'
from Phoenix_project.core.schemas.data_schema import MarketEvent, TickerData
# 修复：将 'ai.data_adapter' 转换为 'Phoenix_project.ai.data_adapter'
from Phoenix_project.ai.data_adapter import DataAdapter
# --- [修复结束] ---


@pytest.fixture
def mock_dependencies():
    """Mocks all dependencies for DataManager."""
    mock_config = {
        "data_manager": {
            "temporal_db_url": "mock_db_url",
            "feature_store_path": "mock_fs_path",
            "cold_storage_config": {"type": "s3", "bucket": "mock-bucket"}
        }
    }
    mock_logger = MagicMock()
    mock_adapter = MagicMock(spec=DataAdapter)
    mock_temporal_db = MagicMock()
    mock_feature_store = MagicMock()
    mock_cold_storage = MagicMock()
    
    return {
        "config": mock_config,
        "logger": mock_logger,
        "adapter": mock_adapter,
        "temporal_db": mock_temporal_db,
        "feature_store": mock_feature_store,
        "cold_storage": mock_cold_storage
    }

@pytest.fixture
def data_manager(mock_dependencies):
    """Fixture to create a DataManager with mocked dependencies."""
    
    # We patch the internals that DataManager tries to create
    with patch("Phoenix_project.data_manager.TemporalDBClient", return_value=mock_dependencies["temporal_db"]), \
         patch("Phoenix_project.data_manager.FeatureStore", return_value=mock_dependencies["feature_store"]), \
         patch("Phoenix_project.data_manager.S3Client", return_value=mock_dependencies["cold_storage"]):
        
        dm = DataManager(
            config=mock_dependencies["config"],
            logger=mock_dependencies["logger"],
            adapter=mock_dependencies["adapter"]
        )
    return dm

# Sample data
NOW = datetime.utcnow()
SAMPLE_MARKET_EVENT_RAW = {
    "source": "manual_test",
    "timestamp": NOW.isoformat(),
    "content": "This is a test event.",
    "metadata": {}
}
SAMPLE_MARKET_EVENT_ADAPTED = MarketEvent(
    id="event_123",
    source="manual_test",
    timestamp=NOW,
    content="This is a test event."
)
SAMPLE_TICKER_DATA = TickerData(
    symbol="AAPL",
    timestamp=NOW,
    open=150.0, high=151.0, low=149.0, close=150.5, volume=1000
)


def test_data_manager_process_event(data_manager, mock_dependencies):
    """
    Tests the `process_event` workflow.
    """
    mock_adapter = mock_dependencies["adapter"]
    mock_temporal_db = mock_dependencies["temporal_db"]
    
    # 1. Mock the adapter's output
    mock_adapter.standardize_event.return_value = SAMPLE_MARKET_EVENT_ADAPTED
    
    # 2. Call the method
    result_event = data_manager.process_event(SAMPLE_MARKET_EVENT_RAW)
    
    # 3. Verify adapter was called
    mock_adapter.standardize_event.assert_called_once_with(SAMPLE_MARKET_EVENT_RAW)
    
    # 4. Verify temporal DB was called
    mock_temporal_db.insert_market_event.assert_called_once_with(SAMPLE_MARKET_EVENT_ADAPTED)
    
    # 5. Verify the correct event is returned
    assert result_event == SAMPLE_MARKET_EVENT_ADAPTED

def test_data_manager_get_context_data(data_manager, mock_dependencies):
    """
    Tests the `get_context_data` workflow.
    """
    mock_temporal_db = mock_dependencies["temporal_db"]
    mock_feature_store = mock_dependencies["feature_store"]
    
    # 1. Mock the DB/Store outputs
    mock_temporal_db.query_ticker_data.return_value = [SAMPLE_TICKER_DATA]
    mock_temporal_db.query_market_events.return_value = [SAMPLE_MARKET_EVENT_ADAPTED]
    mock_feature_store.get_features.return_value = pd.DataFrame({"rsi": [50]}) # Mock
    
    # 2. Call the method
    symbols = ["AAPL"]
    time_window = timedelta(days=1)
    context_data = data_manager.get_context_data(symbols, time_window)
    
    # 3. Verify dependencies were called
    mock_temporal_db.query_ticker_data.assert_called_once_with(symbols, time_window)
    mock_temporal_db.query_market_events.assert_called_once_with(symbols, time_window)
    # mock_feature_store.get_features.assert_called_once() # Call depends on logic
    
    # 4. Verify output structure
    assert "ticker_data" in context_data
    assert "market_events" in context_data
    assert "features" in context_data
    assert context_data["ticker_data"]["AAPL"][0] == SAMPLE_TICKER_DATA
    assert context_data["market_events"][0] == SAMPLE_MARKET_EVENT_ADAPTED

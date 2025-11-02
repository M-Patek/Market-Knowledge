import pytest
import pandas as pd
from pydantic import ValidationError

# 修正：[FIX-ImportError]
# 将所有 `..` 相对导入更改为从项目根目录开始的绝对导入，
# 以匹配 `conftest.py` 设置的 sys.path 约定。
from data_manager import DataManager
from core.pipeline_state import PipelineState
from core.schemas.config_schema import StrategyConfig

# Mock configuration based on config_schema.py
# This is required for DataManager(config) initialization
MOCK_CONFIG_DICT = {
    "start_date": "2020-01-01",
    "end_date": "2023-01-01",
    "asset_universe": ["AAPL", "MSFT"],
    "market_breadth_tickers": ["^VIX"],
    "log_level": "INFO",
    "data_sources": {
        "priority": ["yfinance"],
        "providers": {
            "yfinance": {"api_key_env_var": "NOT_USED"},
            "alpha_vantage": {"api_key_env_var": "AV_API_KEY"}
        },
        "network": {
            "user_agent": "PhoenixTest",
            "proxy": {},
            "request_timeout": 30,
            "retry_attempts": 3,
            "retry_backoff_factor": 1
        },
        "health_probes": {
            "failure_threshold": 3,
            "cooldown_minutes": 10
        }
    },
    "sma_period": 20,
    "rsi_period": 14,
    "rsi_overbought_threshold": 70,
    "opportunity_score_threshold": 70,
    "vix_high_threshold": 30,
    "vix_low_threshold": 15,
    "capital_modifier_high_vix": 0.5,
    "capital_modifier_normal_vix": 1.0,
    "capital_modifier_low_vix": 1.2,
    "initial_cash": 1000000,
    "commission_rate": 0.001,
    "max_total_allocation": 0.9,
    "ai_ensemble_config": {
        "enable": True,
        "config_file_path": "config/ai/ensemble_config.yaml"
    },
    "ai_mode": "hybrid",
    "walk_forward": {
        "start_date": "2020-01-01",
        "end_date": "2023-12-31",
        "training_days": 365,
        "validation_days": 90,
        "step_days": 90
    },
    "execution_model": {
        "impact_coefficient": 0.1,
        "max_volume_share": 0.1,
        "min_trade_notional": 1000
    },
    "position_sizer": {
        "method": "volatility_parity",
        "parameters": {"lookback_period": 20}
    },
    "optimizer": {
        "study_name": "phoenix_strategy_opt",
        "n_trials": 100,
        "parameters": {
            "sma_period": [10, 50],
            "rsi_period": [10, 30]
        }
    },
    "observability": {
        "metrics_port": 8001
    },
    "audit": {
        "s3_bucket_name": "phoenix-audit-logs"
    },
    "data_manager": {
        "cache_dir": "test_cache"
    }
}

# Validate the mock config against the schema
try:
    MOCK_CONFIG = StrategyConfig(**MOCK_CONFIG_DICT)
except ValidationError as e:
    print("FATAL: Mock config is invalid!")
    print(e)
    pytest.fail("Mock config does not match StrategyConfig schema.")


@pytest.fixture
def data_manager():
    """Fixture to create a DataManager instance for testing."""
    # 修正：[FIX-TypeError-DataManager]
    # 匹配 data_manager.py 中已修正的构造函数
    state = PipelineState()
    return DataManager(
        config=MOCK_CONFIG_DICT, 
        pipeline_state=state, 
        cache_dir="test_cache"
    )

def test_data_manager_init(data_manager):
    """Tests if the DataManager initializes correctly."""
    assert data_manager is not None
    assert data_manager.config == MOCK_CONFIG_DICT.get('data_manager', {})
    assert data_manager.cache_dir == "test_cache"

@pytest.mark.skip(reason="Requires live yfinance API call")
def test_get_historical_data_live(data_manager):
    """
    Tests fetching real data from yfinance.
    This test is skipped by default to avoid network dependency.
    """
    symbol = "AAPL"
    start_date = pd.Timestamp("2022-01-01")
    end_date = pd.Timestamp("2022-01-10")
    
    data = data_manager.get_historical_data(symbol, start_date, end_date)
    
    assert data is not None
    assert not data.empty
    assert isinstance(data, pd.DataFrame)
    assert data.index.min() >= start_date
    assert data.index.max() <= end_date
    assert list(data.columns) == ['open', 'high', 'low', 'close', 'volume']

def test_get_historical_data_mocked(data_manager, mocker):
    """Tests fetching data when yfinance is mocked."""
    
    # 1. Create mock data
    mock_df = pd.DataFrame({
        'Open': [100, 101],
        'High': [102, 102],
        'Low': [99, 100],
        'Close': [101, 101],
        'Volume': [1000, 1200],
        'Dividends': [0, 0],
        'Stock Splits': [0, 0]
    }, index=pd.to_datetime(['2022-01-03', '2022-01-04']))
    # yfinance returns tz-aware, localize to UTC
    mock_df.index = mock_df.index.tz_localize('UTC')

    # 2. Mock the yf.Ticker().history() call
    mock_ticker = mocker.MagicMock()
    mock_ticker.history.return_value = mock_df
    mocker.patch("yfinance.Ticker", return_value=mock_ticker)
    
    # 3. Define dates
    symbol = "MSFT"
    start_date = pd.Timestamp("2022-01-01")
    end_date = pd.Timestamp("2022-01-10")

    # 4. Call the function
    data = data_manager.get_historical_data(symbol, start_date, end_date)
    
    # 5. Assertions
    assert data is not None
    assert not data.empty
    assert list(data.columns) == ['open', 'high', 'low', 'close', 'volume']
    assert data.loc['2022-01-03']['close'] == 101
    # Check that yf.Ticker().history was called correctly
    mock_ticker.history.assert_called_with(start='2022-01-01', end='2022-01-11')
    
    # 6. Test caching
    # Call again
    data_cached = data_manager.get_historical_data(symbol, start_date, end_date)
    # yf.Ticker().history should NOT be called again
    assert mock_ticker.history.call_count == 1
    pd.testing.assert_frame_equal(data, data_cached)

def test_data_manager_empty_response(data_manager, mocker):
    """Tests that an empty DataFrame is returned on yfinance failure."""
    
    # 1. Mock yf.Ticker().history() to return an empty DF
    mock_ticker = mocker.MagicMock()
    mock_ticker.history.return_value = pd.DataFrame()
    mocker.patch("yfinance.Ticker", return_value=mock_ticker)
    
    # 2. Call the function
    data = data_manager.get_historical_data(
        "FAIL", 
        pd.Timestamp("2022-01-01"), 
        pd.Timestamp("2022-01-10")
    )
    
    # 3. Assertions
    assert data is not None
    assert data.empty
    assert isinstance(data, pd.DataFrame)

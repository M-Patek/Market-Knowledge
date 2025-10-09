# tests/test_data_manager.py

import pytest
import pandas as pd
from datetime import date, timedelta, datetime
import os
import shutil
from unittest.mock import AsyncMock

from phoenix_project import DataManager, StrategyConfig

@pytest.fixture
def mock_config():
    """Provides a mock StrategyConfig for DataManager testing."""
    config_dict = {
        "start_date": "2023-01-01", "end_date": "2023-01-10",
        "asset_universe": ["TEST_TICKER"],
        "market_breadth_tickers": ["SPY"],
        "log_level": "DEBUG",
        "data_sources": {
            "priority": ["mock_provider_1", "mock_provider_2", "yfinance"],
            "providers": {
                "mock_provider_1": {"api_key_env_var": "FAKE_KEY_1"},
                "mock_provider_2": {"api_key_env_var": "FAKE_KEY_2"},
                 "alpha_vantage": {"api_key_env_var": "FAKE_KEY_AV"},
                 "twelvedata": {"api_key_env_var": "FAKE_KEY_TD"},
            },
            "network": {
                "user_agent": "test", 
                "proxy": {"enabled": False},
                "request_timeout": 5,
                "retry_attempts": 1,
                "retry_backoff_factor": 1
            },
            "health_probes": {"failure_threshold": 2, "cooldown_minutes": 5}
        },
        # Add other required fields with dummy values to satisfy Pydantic validation
        "sma_period": 50, "rsi_period": 14, "rsi_overbought_threshold": 70.0,
        "opportunity_score_threshold": 55.0, "vix_high_threshold": 30.0,
        "vix_low_threshold": 20.0, "capital_modifier_high_vix": 0.5,
        "capital_modifier_normal_vix": 0.9, "capital_modifier_low_vix": 1.0,
        "initial_cash": 100000, "commission_rate": 0.001,
        "gemini_config": {"enable": False, "mode": "mock", "api_key_env_var": "FAKE_KEY", "model_name": "fake", "prompts": {}, "audit_log_retention_days": 1, "max_concurrent_requests": 1},
        "ai_mode": "off", "walk_forward": {}, "max_total_allocation": 1.0,
        "execution_model": {"impact_coefficient": 0.1, "max_volume_share": 0.25, "min_trade_notional": 1.0},
        "position_sizer": {"method": "fixed_fraction", "parameters": {"fraction_per_position": 0.1}},
        "optimizer": {"study_name": "test", "n_trials": 1, "parameters": {}},
        "observability": {"metrics_port": 8002},
        "audit": {"s3_bucket_name": "none"}
    }
    return StrategyConfig(**config_dict)

@pytest.fixture
def data_manager(mock_config, tmp_path):
    """Provides a DataManager instance with a temporary cache directory."""
    cache_dir = tmp_path / "data_cache"
    cache_dir.mkdir()
    # Mock environment variables
    os.environ["FAKE_KEY_AV"] = "key_av"
    os.environ["FAKE_KEY_TD"] = "key_td"
    yield DataManager(config=mock_config, cache_dir=str(cache_dir))
    # Teardown: remove the temp cache dir
    shutil.rmtree(cache_dir)

def create_mock_dataframe(ticker: str, start="2023-01-01", end="2023-01-10"):
    """Creates a sample pandas DataFrame for mocking API responses."""
    dates = pd.to_datetime(pd.date_range(start=start, end=end))
    if dates.empty:
        return pd.DataFrame()
    df = pd.DataFrame({
        'Open': [100] * len(dates), 'High': [105] * len(dates),
        'Low': [99] * len(dates), 'Close': [102] * len(dates),
        'Volume': [10000] * len(dates)
    }, index=dates)
    df.columns = pd.MultiIndex.from_product([df.columns, [ticker]])
    return df

@pytest.mark.asyncio
async def test_cache_miss_and_successful_fetch(data_manager, mocker):
    """Tests a cache miss, followed by a successful fetch from the primary provider."""
    mock_df = create_mock_dataframe("TEST_TICKER")
    mocker.patch.object(data_manager, '_fetch_from_alpha_vantage', new_callable=AsyncMock, return_value=mock_df)
    
    # Change provider names to match the actual methods for patching
    data_manager.ds_config.priority = ["alpha_vantage"]

    result_df = await data_manager.get_asset_universe_data()

    assert result_df is not None
    assert not result_df.empty
    assert "TEST_TICKER" in result_df.columns.get_level_values(1)
    data_manager._fetch_from_alpha_vantage.assert_called_once()

@pytest.mark.asyncio
async def test_cache_hit(data_manager, mocker):
    """Tests that data is loaded from cache if it exists and is up-to-date."""
    # First, run a successful fetch to populate the cache
    mock_df = create_mock_dataframe("TEST_TICKER")
    mocker.patch.object(data_manager, '_fetch_from_alpha_vantage', new_callable=AsyncMock, return_value=mock_df)
    data_manager.ds_config.priority = ["alpha_vantage"]
    await data_manager.get_asset_universe_data()
    data_manager._fetch_from_alpha_vantage.assert_called_once() # Should be called once to cache
    
    # Now, run it again. The mock should not be called a second time.
    result_df = await data_manager.get_asset_universe_data()
    assert result_df is not None
    data_manager._fetch_from_alpha_vantage.assert_called_once() # Still once

@pytest.mark.asyncio
async def test_provider_failure_and_fallback(data_manager, mocker):
    """Tests that the manager falls back to the second provider if the first fails."""
    mock_df = create_mock_dataframe("TEST_TICKER")
    mocker.patch.object(data_manager, '_fetch_from_alpha_vantage', new_callable=AsyncMock, side_effect=Exception("API Failure"))
    mocker.patch.object(data_manager, '_fetch_from_twelvedata', new_callable=AsyncMock, return_value=mock_df)

    data_manager.ds_config.priority = ["alpha_vantage", "twelvedata"]

    result_df = await data_manager.get_asset_universe_data()

    assert result_df is not None
    assert "TEST_TICKER" in result_df.columns.get_level_values(1)
    data_manager._fetch_from_alpha_vantage.assert_called_once()
    data_manager._fetch_from_twelvedata.assert_called_once()

@pytest.mark.asyncio
async def test_circuit_breaker_opens_and_skips_provider(data_manager, mocker):
    """Tests that the circuit breaker opens after consecutive failures and skips the provider."""
    mock_df = create_mock_dataframe("TEST_TICKER")
    mocker.patch.object(data_manager, '_fetch_from_alpha_vantage', new_callable=AsyncMock, side_effect=Exception("API Failure"))
    mocker.patch.object(data_manager, '_fetch_from_twelvedata', new_callable=AsyncMock, return_value=mock_df)
    
    data_manager.ds_config.priority = ["alpha_vantage", "twelvedata"]
    failure_threshold = data_manager.ds_config.health_probes.failure_threshold

    # Trigger failures to open the circuit
    for _ in range(failure_threshold):
        await data_manager.get_asset_universe_data()

    assert data_manager._provider_health["alpha_vantage"]["cooldown_until"] is not None
    assert data_manager._fetch_from_alpha_vantage.call_count == failure_threshold
    
    # Now, make another call. The failing provider should be skipped entirely.
    await data_manager.get_asset_universe_data()
    assert data_manager._fetch_from_alpha_vantage.call_count == failure_threshold # Count should not increase

@pytest.mark.asyncio
async def test_incremental_update(data_manager, mocker):
    """Tests that only new data is fetched if the cache is partial."""
    # 1. Populate cache with initial data
    initial_end_date = date(2023, 1, 5)
    initial_df = create_mock_dataframe("TEST_TICKER", end=initial_end_date.isoformat())
    mocker.patch.object(data_manager, '_fetch_from_alpha_vantage', new_callable=AsyncMock, return_value=initial_df)
    data_manager.ds_config.priority = ["alpha_vantage"]
    data_manager.config.end_date = initial_end_date
    await data_manager.get_asset_universe_data()
    
    # 2. Change end_date and mock the fetch for the new, incremental data
    new_end_date = date(2023, 1, 10)
    data_manager.config.end_date = new_end_date
    incremental_df = create_mock_dataframe("TEST_TICKER", start=(initial_end_date + timedelta(days=1)).isoformat(), end=new_end_date.isoformat())
    data_manager._fetch_from_alpha_vantage.return_value = incremental_df
    
    # 3. Run again and check results
    final_df = await data_manager.get_asset_universe_data()
    
    # Assert that the fetch was only called for the incremental part
    data_manager._fetch_from_alpha_vantage.assert_called_with(
        tickers=['TEST_TICKER'], 
        start=initial_end_date + timedelta(days=1), 
        end=new_end_date
    )
    assert final_df is not None
    assert final_df.index.min().date() == date(2023, 1, 1)
    assert final_df.index.max().date() == new_end_date

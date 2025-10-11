# tests/test_data_manager.py

import pytest
import pandas as pd
from datetime import date, timedelta, datetime
import os
import shutil
from unittest.mock import AsyncMock, patch

from phoenix_project import DataManager, StrategyConfig

@pytest.fixture
def mock_config():
    """Provides a mock StrategyConfig for DataManager testing."""
    config_dict = {
        "start_date": "2023-01-01", "end_date": "2023-01-10",
        "asset_universe": ["TICKER_A", "TICKER_B"],
        "market_breadth_tickers": ["SPY", "QQQ"],
        "log_level": "DEBUG",
        "data_sources": {
            "priority": ["alpha_vantage", "twelvedata", "yfinance"],
            "providers": {
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
        "sma_period": 10, "rsi_period": 10, "rsi_overbought_threshold": 70,
        "opportunity_score_threshold": 50, "vix_high_threshold": 30,
        "vix_low_threshold": 20, "capital_modifier_high_vix": 0.5,
        "capital_modifier_normal_vix": 0.9, "capital_modifier_low_vix": 1.0,
        "initial_cash": 100000, "commission_rate": 0.001, "max_total_allocation": 1.0,
        "ai_ensemble_config": {"enable": False, "config_file_path": "fake.yaml"},
        "ai_mode": "off", "walk_forward": {},
        "execution_model": {"impact_coefficient": 0.1, "max_volume_share": 0.25, "min_trade_notional": 100},
        "position_sizer": {"method": "fixed_fraction", "parameters": {"fraction_per_position": 0.1}},
        "optimizer": {"study_name": "test", "n_trials": 1, "parameters": {}},
        "observability": {"metrics_port": 8002},
        "audit": {"s3_bucket_name": "none"}
    }
    return StrategyConfig(**config_dict)

@pytest.fixture
def data_manager(mock_config, tmp_path) -> DataManager:
    """Provides a DataManager instance with a temporary cache directory."""
    cache_dir = tmp_path / "data_cache"
    cache_dir.mkdir()
    os.environ["FAKE_KEY_AV"] = "key_av_xyz"
    os.environ["FAKE_KEY_TD"] = "key_td_xyz"
    yield DataManager(config=mock_config, cache_dir=str(cache_dir))
    shutil.rmtree(cache_dir)

def create_mock_dataframe(ticker: str, start="2023-01-01", end="2023-01-10"):
    """Creates a sample pandas DataFrame for mocking API responses."""
    dates = pd.to_datetime(pd.date_range(start=start, end=end))
    df = pd.DataFrame({
        'Open': [100] * len(dates), 'High': [105] * len(dates),
        'Low': [99] * len(dates), 'Close': [102] * len(dates),
        'Volume': [10000] * len(dates)
    }, index=dates)
    df.index.name = 'Date'
    # Match the column structure of the real data
    df.columns = pd.MultiIndex.from_product([df.columns, [ticker]])
    return df

@pytest.mark.asyncio
async def test_concurrent_fetch_cache_miss(data_manager, mock_config):
    """Tests concurrent cache misses, followed by successful fetches from the primary provider."""
    with patch.object(data_manager, '_fetch_from_alpha_vantage', new_callable=AsyncMock) as mock_fetch:
        # Configure the mock to return a different DataFrame for each ticker
        mock_fetch.side_effect = lambda tickers, start, end: create_mock_dataframe(tickers[0], start, end)
        
        result_df = await data_manager.get_asset_universe_data()

        assert result_df is not None
        assert not result_df.empty
        assert "TICKER_A" in result_df.columns.get_level_values(1)
        assert "TICKER_B" in result_df.columns.get_level_values(1)
        assert mock_fetch.call_count == len(mock_config.asset_universe)

@pytest.mark.asyncio
async def test_concurrent_cache_hit(data_manager):
    """Tests that data is loaded from cache if it exists and is up-to-date."""
    with patch.object(data_manager, '_fetch_from_alpha_vantage', new_callable=AsyncMock) as mock_fetch:
        mock_fetch.side_effect = lambda tickers, start, end: create_mock_dataframe(tickers[0], start, end)
        
        # 1. Run once to populate the cache
        await data_manager.get_asset_universe_data()
        assert mock_fetch.call_count == 2

        # 2. Run again. The mock should not be called a second time.
        result_df = await data_manager.get_asset_universe_data()
        assert result_df is not None
        assert mock_fetch.call_count == 2 # Should still be 2

@pytest.mark.asyncio
async def test_concurrent_provider_failure_and_fallback(data_manager):
    """Tests that the manager falls back for each ticker independently."""
    with patch.object(data_manager, '_fetch_from_alpha_vantage', new_callable=AsyncMock, side_effect=Exception("API Failure")), \
         patch.object(data_manager, '_fetch_from_twelvedata', new_callable=AsyncMock) as mock_fallback:
        
        mock_fallback.side_effect = lambda tickers, start, end: create_mock_dataframe(tickers[0], start, end)

        result_df = await data_manager.get_asset_universe_data()

        assert result_df is not None
        assert "TICKER_A" in result_df.columns.get_level_values(1)
        assert "TICKER_B" in result_df.columns.get_level_values(1)
        assert data_manager._fetch_from_alpha_vantage.call_count == 2
        assert mock_fallback.call_count == 2

@pytest.mark.asyncio
async def test_circuit_breaker_opens_and_skips_provider(data_manager):
    """Tests that the circuit breaker opens after consecutive failures and skips the provider."""
    failure_threshold = data_manager.ds_config。health_probes。failure_threshold
    
    with patch.object(data_manager, '_fetch_from_alpha_vantage', new_callable=AsyncMock, side_effect=Exception("API Failure")), \
         patch.object(data_manager, '_fetch_from_twelvedata', new_callable=AsyncMock) as mock_fallback:

        mock_fallback.side_effect = lambda tickers, start, end: create_mock_dataframe(tickers[0], start, end)
        
        # Trigger failures to open the circuit
        await data_manager.get_asset_universe_data()
        
        # Since two tickers fail concurrently, the failure count should now be 2, opening the breaker
        assert data_manager._provider_health["alpha_vantage"]["failures"] == failure_threshold
        assert data_manager._provider_health["alpha_vantage"]["cooldown_until"] is not None
        
        # Now, make another call. The failing provider should be skipped entirely.
        await data_manager.get_asset_universe_data()
        # Call count should not increase because the provider is in cooldown
        assert data_manager._fetch_from_alpha_vantage.call_count == failure_threshold

@pytest.mark.asyncio
async def test_incremental_update(data_manager: DataManager):
    """Tests that only new data is fetched if the cache is partial for a single ticker."""
    with patch.object(data_manager, '_fetch_from_alpha_vantage', new_callable=AsyncMock) as mock_fetch:
        # 1. Populate cache with initial data for TICKER_A
        initial_end_date = date(2023， 1, 5)
        data_manager.config。end_date = initial_end_date
        mock_fetch.side_effect = lambda tickers, start, end: create_mock_dataframe(tickers[0], start, end)
        await data_manager.get_asset_universe_data()
        
        # 2. Change end_date and mock the fetch for the new, incremental data
        new_end_date = date(2023， 1, 10)
        data_manager.config。end_date = new_end_date
        
        final_df = await data_manager.get_asset_universe_data()
        
        # Assert that the fetch was only called for the incremental part for both tickers
        # The mock was called twice initially, and twice for the incremental update.
        assert mock_fetch.call_count == 4
        
        # Check call arguments for the incremental fetch of TICKER_A
        # Note: with asyncio.gather, order is not guaranteed, so we check one of the last two calls
        incremental_call_args_A = mock_fetch.call_args_list[-2].args
        incremental_call_args_B = mock_fetch.call_args_list[-1].args

        # Find the call for TICKER_A
        call_for_A = None
        if incremental_call_args_A[0] == ['TICKER_A']:
            call_for_A = incremental_call_args_A
        elif incremental_call_args_B[0] == ['TICKER_A']:
            call_for_A = incremental_call_args_B

        assert call_for_A is not None
        assert call_for_A[1] == initial_end_date + timedelta(days=1)
        assert call_for_A[2] == new_end_date

        assert final_df is not None
        assert final_df.index.min().date() == date(2023, 1, 1)
        assert final_df.index.max().date() == new_end_date

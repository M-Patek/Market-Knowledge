# tests/test_strategy_data_handler.py

import pytest
import pandas as pd
from datetime import date
import backtrader as bt

from phoenix_project import StrategyConfig
from cognitive.engine import CognitiveEngine
from strategy_handler import StrategyDataHandler, DailyDataPacket

@pytest.fixture
def mock_config():
    """Provides a mock StrategyConfig for testing."""
    config_dict = {
        "start_date": "2022-01-01", "end_date": "2023-01-01",
        "asset_universe": ["SPY"], "market_breadth_tickers": ["SPY"],
        "sma_period": 1, "rsi_period": 2, "rsi_overbought_threshold": 70.0,
        "opportunity_score_threshold": 55.0, "vix_high_threshold": 30.0,
        "vix_low_threshold": 20.0, "capital_modifier_high_vix": 0.5,
        "capital_modifier_normal_vix": 0.9, "capital_modifier_low_vix": 1.0,
        "initial_cash": 100000, "commission_rate": 0.001,
        "log_level": "INFO", "ai_ensemble_config": {"enable": False, "config_file_path": "fake"},
        "data_sources": {"priority": [], "providers": {}, "network": {"retry_attempts": 1, "retry_backoff_factor": 1, "request_timeout": 5, "user_agent": "test", "proxy": {"enabled": False}}, "health_probes": {"failure_threshold": 1, "cooldown_minutes": 1}},
        "ai_mode": "off", "walk_forward": {}, "max_total_allocation": 1.0,
        "execution_model": {"impact_coefficient": 0.1, "max_volume_share": 0.25, "min_trade_notional": 100.0},
        "position_sizer": {"method": "fixed_fraction", "parameters": {"fraction_per_position": 0.1}},
        "optimizer": {"study_name": "test", "n_trials": 1, "parameters": {}},
        "observability": {"metrics_port": 8001},
        "audit": {"s3_bucket_name": "none"}
    }
    return StrategyConfig(**config_dict)

@pytest.fixture
def mock_strategy_instance(mock_config):
    """Creates a Cerebro instance and returns an initialized strategy for testing."""
    cerebro = bt.Cerebro(stdstats=False)
    
    # Create dummy data that allows indicators to calculate
    data_df = pd.DataFrame({
        'open': [100, 101, 102],
        'high': [105, 106, 107],
        'low': [99, 100, 101],
        'close': [102, 103, 104],
        'volume': [10000, 11000, 12000]
    }, index=pd.to_datetime(['2023-01-08', '2023-01-09', '2023-01-10']))
    
    data_feed = bt.feeds.PandasData(dataname=data_df, name='SPY')
    cerebro.adddata(data_feed)
    
    # A dummy strategy class to host the handler
    class TestStrategy(bt.Strategy):
        pass

    cerebro.addstrategy(TestStrategy)
    
    # cerebro.run() returns a list of the strategies
    return cerebro.run()[0]

def test_data_handler_initialization(mock_strategy_instance, mock_config):
    test_date = date(2023, 1, 10)
    vix_data = pd.Series([25.0], index=[pd.to_datetime(test_date)])
    yield_data = pd.Series([3.5], index=[pd.to_datetime(test_date)])
    breadth_data = pd.Series([0.6], index=[pd.to_datetime(test_date)])

    handler = StrategyDataHandler(mock_strategy_instance, mock_config, vix_data, yield_data, breadth_data)
    
    assert handler is not None
    assert 'SPY' in handler.sma_indicators
    assert 'SPY' in handler.rsi_indicators
    assert handler.vix_lookup[test_date] == 25.0

def test_get_daily_data_packet(mock_strategy_instance, mock_config):
    test_date = date(2023, 1, 10)
    vix_data = pd.Series([25.0], index=[pd.to_datetime(test_date)])
    yield_data = pd.Series([3.5], index=[pd.to_datetime(test_date)])
    breadth_data = pd.Series([0.6], index=[pd.to_datetime(test_date)])

    handler = StrategyDataHandler(mock_strategy_instance, mock_config, vix_data, yield_data, breadth_data)
    cognitive_engine = CognitiveEngine(config=mock_config)

    # The mock_strategy_instance from cerebro.run() is already at the last bar
    packet = handler.get_daily_data_packet(cognitive_engine)
    
    assert isinstance(packet, DailyDataPacket)
    assert packet.current_date == test_date
    assert packet.current_vix == 25.0
    assert len(packet.candidate_analysis) == 1
    assert packet.candidate_analysis[0]['ticker'] == 'SPY'
    assert 'opportunity_score' in packet.candidate_analysis[0]

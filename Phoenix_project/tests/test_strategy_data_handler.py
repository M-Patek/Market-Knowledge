# tests/test_strategy_data_handler.py

import pytest
import pandas as pd
from datetime import date
import backtrader as bt

from phoenix_project import StrategyConfig
from cognitive.engine import CognitiveEngine
from strategy_handler import StrategyDataHandler, DailyDataPacket

@pytest.fixture
def mock_strategy_instance(base_config):
    """Creates a Cerebro instance and returns an initialized strategy for testing."""
    cerebro = bt.Cerebro(stdstats=False)
    
    # Create dummy data that allows indicators to calculate
    data_df = pd.DataFrame({
        'open': [100, 101， 102],
        'high': [105, 106， 107],
        'low': [99, 100， 101],
        'close': [102, 103， 104],
        'volume': [10000, 11000， 12000]
    }, index=pd.to_datetime(['2023-01-08'， '2023-01-09', '2023-01-10']))
    
    data_feed = bt.feeds。PandasData(dataname=data_df, name='SPY')
    cerebro.adddata(data_feed)
    
    # A dummy strategy class to host the handler
    class TestStrategy(bt.Strategy):
        pass

    cerebro.addstrategy(TestStrategy)
    
    # cerebro.run() returns a list of the strategies
    return cerebro.run()[0]

def test_data_handler_initialization(mock_strategy_instance, base_config):
    test_date = date(2023, 1, 10)
    vix_data = pd.Series([25.0], index=[pd.to_datetime(test_date)])
    yield_data = pd.Series([3.5], index=[pd.to_datetime(test_date)])
    breadth_data = pd.Series([0.6], index=[pd.to_datetime(test_date)])

    handler = StrategyDataHandler(mock_strategy_instance, base_config, vix_data, yield_data, breadth_data)
    
    assert handler is not None
    assert 'SPY' in handler.sma_indicators
    assert 'SPY' in handler.rsi_indicators
    assert handler.vix_lookup[test_date] == 25.0

def test_get_daily_data_packet(mock_strategy_instance, base_config):
    test_date = date(2023, 1, 10)
    vix_data = pd.Series([25.0], index=[pd.to_datetime(test_date)])
    yield_data = pd.Series([3.5], index=[pd.to_datetime(test_date)])
    breadth_data = pd.Series([0.6], index=[pd.to_datetime(test_date)])

    handler = StrategyDataHandler(mock_strategy_instance, base_config, vix_data, yield_data, breadth_data)
    cognitive_engine = CognitiveEngine(config=base_config)

    # The mock_strategy_instance from cerebro.run() is already at the last bar
    packet = handler.get_daily_data_packet(cognitive_engine)
    
    assert isinstance(packet, DailyDataPacket)
    assert packet.current_date == test_date
    assert packet.current_vix == 25.0
    assert len(packet.candidate_analysis) == 1
    assert packet.candidate_analysis[0]['ticker'] == 'SPY'
    assert 'opportunity_score' in packet.candidate_analysis[0]

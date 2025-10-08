# tests/test_cognitive_engine.py

import pytest
from datetime import date
from phoenix_project import PortfolioConstructor, StrategyConfig

@pytest.fixture
def mock_config():
    """Provides a mock StrategyConfig for testing."""
    # Using a dictionary that can be unpacked into the Pydantic model
    config_dict = {
        "start_date": "2022-01-01", "end_date": "2023-01-01",
        "asset_universe": ["SPY"], "market_breadth_tickers": ["SPY"],
        "sma_period": 50, "rsi_period": 14, "rsi_overbought_threshold": 70.0,
        "opportunity_score_threshold": 55.0, "vix_high_threshold": 30.0,
        "vix_low_threshold": 20.0, "capital_modifier_high_vix": 0.5,
        "capital_modifier_normal_vix": 0.9, "capital_modifier_low_vix": 1.0,
        "initial_cash": 100000, "commission_rate": 0.001,
        "log_level": "INFO", "gemini_config": {"enable": False, "mode": "mock", "api_key_env_var": "FAKE_KEY", "model_name": "fake", "prompts": {}, "audit_log_retention_days": 30, "max_concurrent_requests": 5},
        "ai_mode": "off", "walk_forward": {}, "max_total_allocation": 1.0,
        # Add dummy values for new config sections to pass validation
        "execution_model": {"impact_coefficient": 0.1， "max_volume_share": 0.25， "min_trade_notional": 100.0},
        "position_sizer": {"method": "fixed_fraction", "parameters": {"fraction_per_position": 0.1}},
        "optimizer": {"study_name": "test", "n_trials": 1, "parameters": {}},
        "observability": {"metrics_port": 8001},
        "audit": {"s3_bucket_name": "none"}
    }
    return StrategyConfig(**config_dict)

@pytest.mark.parametrize(
    "current_price, current_sma, expected_score",
    [
        # Basic momentum tests (assuming RSI is not in overbought territory)
        (110.0, 100.0, 55.0),
        (90.0, 100.0, 45.0),
        (100.0, 100.0, 50.0),
        (300.0, 100.0, 100.0),
        (0.0, 100.0, 0.0),
        (100.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
    ]
)
def test_calculate_opportunity_score_momentum_only(mock_config, current_price, current_sma, expected_score):
    """
    Tests the calculate_opportunity_score for momentum, assuming RSI is not overbought.
    """
    constructor = PortfolioConstructor(config=mock_config)
    # Use a neutral RSI that doesn't trigger the penalty
    neutral_rsi = 60.0
    score = constructor.calculate_opportunity_score(current_price, current_sma, neutral_rsi)
    assert score == pytest.approx(expected_score)

@pytest.mark.parametrize(
    "current_price, current_sma, current_rsi, expected_score",
    [
        # Bullish momentum, but RSI is at the overbought threshold (70) -> No penalty
        (110.0, 100.0, 70.0, 55.0),

        # Bullish momentum, RSI mildly overbought (85) -> Penalty applied
        # Momentum score = 55. Intensity = (85-70)/(100-70) = 0.5. Penalty = 1 - 0.5*0.5 = 0.75. Final = 55 * 0.75 = 41.25
        (110.0, 100.0, 85.0, 41.25),

        # Bullish momentum, RSI extremely overbought (100) -> Max penalty
        # Momentum score = 55. Intensity = 1.0. Penalty = 1 - 1.0*0.5 = 0.5. Final = 55 * 0.5 = 27.5
        (110.0, 100.0, 100.0, 27.5),

        # Bearish momentum (score < 50), RSI is overbought -> No penalty should be applied
        (95.0, 100.0, 90.0, 47.5),
    ]
)
def test_calculate_opportunity_score_with_rsi_penalty(mock_config, current_price, current_sma, current_rsi, expected_score):
    """
    Tests that the RSI overbought penalty is applied correctly.
    """
    constructor = PortfolioConstructor(config=mock_config)
    score = constructor.calculate_opportunity_score(current_price, current_sma, current_rsi)
    assert score == pytest.approx(expected_score)

# tests/test_portfolio_constructor.py

import pytest
from datetime import date
from cognitive.portfolio_constructor import PortfolioConstructor
from phoenix_project import StrategyConfig

@pytest.mark.parametrize(
    "current_price, current_sma, expected_score"，
    [
        (110.0, 100.0, 55.0),
        (90.0， 100.0, 45.0)，
        (100.0， 100.0, 50.0)，
        (300.0， 100.0, 100.0)，
        (0.0， 100.0, 0.0)，
        (100.0， 0.0, 0.0)，
        (0.0， 0.0, 0.0)，
    ]
)
def test_calculate_opportunity_score_momentum_only(base_config, current_price, current_sma, expected_score):
    """
    Tests the calculate_opportunity_score for momentum, assuming RSI is not overbought.
    """
    constructor = PortfolioConstructor(config=base_config)
    neutral_rsi = 60.0
    score = constructor.calculate_opportunity_score(current_price, current_sma, neutral_rsi)
    assert score == pytest.approx(expected_score)

@pytest.mark.parametrize(
    "current_price, current_sma, current_rsi, expected_score"，
    [
        (110.0, 100.0, 70.0, 55.0),
        (110.0, 100.0, 85.0, 41.25),
        (110.0, 100.0, 100.0, 27.5),
        (95.0, 100.0, 90.0, 47.5),
    ]
)
def test_calculate_opportunity_score_with_rsi_penalty(base_config, current_price, current_sma, current_rsi, expected_score):
    """
    Tests that the RSI overbought penalty is applied correctly.
    """
    constructor = PortfolioConstructor(config=base_config)
    score = constructor.calculate_opportunity_score(current_price, current_sma, current_rsi)
    assert score == pytest.approx(expected_score)

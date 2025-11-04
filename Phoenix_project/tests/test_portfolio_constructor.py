# tests/test_portfolio_constructor.py
import pytest
from unittest.mock import MagicMock, patch

# --- [修复] ---
# 修复：将 'cognitive.portfolio_constructor' 转换为 'Phoenix_project.cognitive.portfolio_constructor'
from Phoenix_project.cognitive.portfolio_constructor import PortfolioConstructor
# 修复：将 'sizing.fixed_fraction' 转换为 'Phoenix_project.sizing.fixed_fraction'
from Phoenix_project.sizing.fixed_fraction import FixedFractionSizer
# 修复：将 'sizing.volatility_parity' 转换为 'Phoenix_project.sizing.volatility_parity'
from Phoenix_project.sizing.volatility_parity import VolatilityParitySizer
# --- [修复结束] ---


@pytest.fixture
def sizer_config():
    """Provides the configuration for position sizers."""
    return {
        "position_sizers": [
            {
                "name": "fixed_fraction",
                "type": "FixedFraction",
                "params": {"fraction_per_position": 0.05} # 5% per asset
            },
            {
                "name": "volatility_parity",
                "type": "VolatilityParity",
                "params": {"volatility_period": 20}
            }
        ]
    }

@pytest.fixture
def portfolio_constructor(sizer_config):
    """Fixture to create a PortfolioConstructor."""
    mock_logger = MagicMock()
    # We patch the logger inside the constructor
    with patch("Phoenix_project.cognitive.portfolio_constructor.logger", mock_logger):
        constructor = PortfolioConstructor(sizer_config, mock_logger)
    return constructor

@pytest.fixture
def trade_candidates():
    """Sample trade candidates from FusionResult."""
    return [
        {"ticker": "AAPL", "reasoning": "Strong buy", "confidence": 0.9, "volatility": 0.2},
        {"ticker": "GOOG", "reasoning": "Weak buy", "confidence": 0.6, "volatility": 0.4},
        {"ticker": "MSFT", "reasoning": "Hold", "confidence": 0.5, "volatility": 0.1}, # Should be filtered
        {"ticker": "TSLA", "reasoning": "Strong buy", "confidence": 0.85, "volatility": 0.8}
    ]

# --- Test Initialization ---

def test_constructor_initialization(portfolio_constructor, sizer_config):
    """
    Tests that the PortfolioConstructor correctly initializes its
    position sizer strategies from the config.
    """
    assert "fixed_fraction" in portfolio_constructor.sizers
    assert "volatility_parity" in portfolio_constructor.sizers
    
    assert isinstance(portfolio_constructor.sizers["fixed_fraction"], FixedFractionSizer)
    assert isinstance(portfolio_constructor.sizers["volatility_parity"], VolatilityParitySizer)
    
    # Check if params were passed
    assert portfolio_constructor.sizers["fixed_fraction"].fraction == 0.05

def test_constructor_handles_invalid_sizer_type(sizer_config):
    """
    Tests that initialization logs an error for an unknown sizer type.
    """
    invalid_config = sizer_config.copy()
    invalid_config["position_sizers"].append(
        {"name": "invalid", "type": "UnknownSizer", "params": {}}
    )
    
    mock_logger = MagicMock()
    with patch("Phoenix_project.cognitive.portfolio_constructor.logger", mock_logger):
        PortfolioConstructor(invalid_config, mock_logger)
    
    # Check that an error was logged
    mock_logger.error.assert_called_with("Unknown position sizer type: UnknownSizer")

# --- Test Filtering ---

def test_filter_candidates(portfolio_constructor, trade_candidates):
    """
    Tests the `_filter_candidates` method (e.g., confidence threshold).
    
    (Note: The implementation of `_filter_candidates` in the provided
    file is a simple passthrough. This test assumes a filter exists,
    e.g., confidence > 0.7)
    """
    # Let's mock the internal filter threshold for this test
    portfolio_constructor.confidence_threshold = 0.7
    
    filtered = portfolio_constructor._filter_candidates(trade_candidates)
    
    assert len(filtered) == 2
    assert filtered[0]["ticker"] == "AAPL"
    assert filtered[1]["ticker"] == "TSLA"

# --- Test Sizing ---

def test_construct_battle_plan_fixed_fraction(portfolio_constructor, trade_candidates):
    """
    Tests the main method using the 'fixed_fraction' sizer.
    """
    # Mock the filter to pass all candidates for sizing
    portfolio_constructor._filter_candidates = MagicMock(return_value=trade_candidates)
    
    # Use the 'fixed_fraction' sizer (5% per asset)
    portfolio_constructor.active_sizer_name = "fixed_fraction"
    
    max_alloc = 0.50 # 50% max total
    
    battle_plan = portfolio_constructor.construct_battle_plan(
        candidates=trade_candidates,
        max_total_allocation=max_alloc
    )
    
    # 4 candidates * 5% = 20% (which is < 50% max)
    assert len(battle_plan) == 4
    assert battle_plan[0]["ticker"] == "AAPL"
    assert battle_plan[0]["capital_allocation_pct"] == 0.05
    assert battle_plan[3]["ticker"] == "TSLA"
    assert battle_plan[3]["capital_allocation_pct"] == 0.05

def test_construct_battle_plan_sizer_scaling(portfolio_constructor, trade_candidates):
    """
    Tests that the sizer correctly scales down if the
    total allocation exceeds the max_total_allocation.
    """
    portfolio_constructor._filter_candidates = MagicMock(return_value=trade_candidates)
    portfolio_constructor.active_sizer_name = "fixed_fraction"
    
    # Set max allocation to 10%
    # The sizer will try 4 * 5% = 20%
    max_alloc = 0.10
    
    battle_plan = portfolio_constructor.construct_battle_plan(
        candidates=trade_candidates,
        max_total_allocation=max_alloc
    )
    
    # The 20% should be scaled down by 0.10 / 0.20 = 0.5
    # Each position should be 0.05 * 0.5 = 0.025
    assert len(battle_plan) == 4
    assert battle_plan[0]["capital_allocation_pct"] == pytest.approx(0.025)
    
    total_alloc = sum(p["capital_allocation_pct"] for p in battle_plan)
    assert total_alloc == pytest.approx(0.10)

def test_construct_battle_plan_volatility_parity(portfolio_constructor, trade_candidates):
    """
    Tests the main method using the 'volatility_parity' sizer.
    """
    # Filter out MSFT (confidence 0.5) and TSLA (volatility 0.8)
    # (Assuming filter logic)
    candidates = [
        {"ticker": "AAPL", "confidence": 0.9, "volatility": 0.2}, # InvVol = 5.0
        {"ticker": "GOOG", "confidence": 0.6, "volatility": 0.4}, # InvVol = 2.5
    ]
    portfolio_constructor._filter_candidates = MagicMock(return_value=candidates)
    
    portfolio_constructor.active_sizer_name = "volatility_parity"
    max_alloc = 0.50 # 50% max total
    
    battle_plan = portfolio_constructor.construct_battle_plan(
        candidates=candidates,
        max_total_allocation=max_alloc
    )
    
    # Total InvVol = 5.0 + 2.5 = 7.5
    # AAPL Weight = (5.0 / 7.5) * 0.50 = (2/3) * 0.50 = 0.333...
    # GOOG Weight = (2.5 / 7.5) * 0.50 = (1/3) * 0.50 = 0.166...
    
    assert len(battle_plan) == 2
    assert battle_plan[0]["ticker"] == "AAPL"
    assert battle_plan[0]["capital_allocation_pct"] == pytest.approx(0.50 * (2/3))
    
    assert battle_plan[1]["ticker"] == "GOOG"
    assert battle_plan[1]["capital_allocation_pct"] == pytest.approx(0.50 * (1/3))
    
    total_alloc = sum(p["capital_allocation_pct"] for p in battle_plan)
    assert total_alloc == pytest.approx(0.50)

# tests/test_cognitive_engine_properties.py
import pytest
from unittest.mock import MagicMock

# --- [修复] ---
# 修复：将 'cognitive.engine' 转换为 'Phoenix_project.cognitive.engine'
from Phoenix_project.cognitive.engine import CognitiveEngine
# 修复：将 'core.pipeline_state' 转换为 'Phoenix_project.core.pipeline_state'
from Phoenix_project.core.pipeline_state import PipelineState
# 修复：将 'core.schemas.fusion_result' 转换为 'Phoenix_project.core.schemas.fusion_result'
from Phoenix_project.core.schemas.fusion_result import FusionResult
# 修复：将 'core.schemas.data_schema' 转换为 'Phoenix_project.core.schemas.data_schema'
from Phoenix_project.core.schemas.data_schema import Signal
# --- [修复结束] ---

# We use the fixtures from conftest.py
# mock_data_manager, mock_risk_manager, mock_portfolio_constructor

@pytest.fixture
def cognitive_engine(mock_data_manager, mock_risk_manager, mock_portfolio_constructor):
    """Fixture to create a CognitiveEngine with mocked dependencies."""
    
    # Mock config
    config = {
        "cognitive_engine": {
            "max_allocation_per_asset": 0.10,
            "max_total_allocation": 0.80,
            "risk_model": "global_stop_loss",
            "portfolio_sizer": "fixed_fraction"
        },
        # Mocks for sizers (used by PortfolioConstructor)
        "position_sizers": [
             {"name": "fixed_fraction", "type": "FixedFraction", "params": {"fraction_per_position": 0.05}}
        ]
    }
    
    # Mock logger
    mock_logger = MagicMock()
    
    engine = CognitiveEngine(
        config=config,
        data_manager=mock_data_manager,
        risk_manager=mock_risk_manager,
        portfolio_constructor=mock_portfolio_constructor,
        logger=mock_logger,
        metrics=MagicMock() # Mock metrics
    )
    return engine

def test_cognitive_engine_generates_signals(cognitive_engine, mock_portfolio_constructor):
    """
    Tests the main `generate_signals` workflow.
    """
    # 1. Create a PipelineState with a FusionResult
    state = PipelineState(event_id="test_event_002")
    state.fusion_result = FusionResult(
        event_id="test_event_002",
        timestamp="2023-01-01T12:00:00Z",
        # This is the key input: candidates
        trade_candidates=[
            {"ticker": "AAPL", "reasoning": "Strong buy", "confidence": 0.9, "volatility": 0.2},
            {"ticker": "GOOG", "reasoning": "Weak buy", "confidence": 0.6, "volatility": 0.3}
        ],
        assessment="Overall bullish"
    )

    # 2. Mock the output of the PortfolioConstructor
    # (which is called by the engine)
    mock_battle_plan = [
        {"ticker": "AAPL", "capital_allocation_pct": 0.05},
        {"ticker": "GOOG", "capital_allocation_pct": 0.05}
    ]
    mock_portfolio_constructor.construct_battle_plan.return_value = mock_battle_plan

    # 3. Run the engine's signal generation
    final_state = cognitive_engine.generate_signals(state)

    # 4. Verify the results
    
    # Check that the portfolio constructor was called correctly
    mock_portfolio_constructor.construct_battle_plan.assert_called_once_with(
        candidates=state.fusion_result.trade_candidates,
        max_total_allocation=0.80 # From config
    )
    
    # Check that the final signals are in the state
    assert final_state.signals is not None
    assert len(final_state.signals) == 2
    
    # Check the Signal objects
    assert isinstance(final_state.signals[0], Signal)
    assert final_state.signals[0].symbol == "AAPL"
    assert final_state.signals[0].signal_type == "BUY" # Assumes conversion logic
    assert final_state.signals[0].strength == 0.05 # Strength = allocation %
    
    assert final_state.signals[1].symbol == "GOOG"
    assert final_state.signals[1].strength == 0.05

def test_cognitive_engine_handles_no_candidates(cognitive_engine, mock_portfolio_constructor):
    """
    Tests that the engine handles a FusionResult with no candidates.
    """
    state = PipelineState(event_id="test_event_003")
    state.fusion_result = FusionResult(
        event_id="test_event_003",
        timestamp="2023-01-01T12:00:00Z",
        trade_candidates=[], # No candidates
        assessment="Neutral"
    )

    final_state = cognitive_engine.generate_signals(state)

    # Ensure portfolio constructor was not called
    mock_portfolio_constructor.construct_battle_plan.assert_not_called()
    
    # Ensure no signals are generated
    assert final_state.signals is not None
    assert len(final_state.signals) == 0

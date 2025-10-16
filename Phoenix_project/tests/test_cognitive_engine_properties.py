# tests/test_cognitive_engine_properties.py
import pytest
from hypothesis import given, strategies as st, settings
from cognitive.engine import CognitiveEngine

# Define a strategy for generating valid, non-zero floats to avoid division by zero
# and restrict extreme values that are unrealistic for financial data.
valid_floats = st.floats(min_value=1e-6, max_value=1e9, allow_nan=False, allow_infinity=False)

@settings(max_examples=500, deadline=None) # Increase test cases
@given(
    current_price=valid_floats,
    sma=valid_floats,
    rsi=st.floats(min_value=-50, max_value=150) # RSI can theoretically go out of 0-100 bounds
)
def test_opportunity_score_invariant(base_config, current_price, sma, rsi):
    """
    Property-Based Test: Ensures the opportunity score is always within [0, 100].
    Hypothesis will generate a wide range of inputs to find edge cases.
    """
    # Arrange
    engine = CognitiveEngine(config=base_config)

    # Act
    score = engine.calculate_opportunity_score(current_price, sma, rsi)

    # Assert
    # This is the "invariant" or "property" we are testing.
    # It must hold true for all generated inputs.
    assert 0.0 <= score <= 100.0, f"Score {score} out of bounds for inputs ({current_price}, {sma}, {rsi})"


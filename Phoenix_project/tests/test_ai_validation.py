# tests/test_ai_validation.py

import pytest
from pydantic import ValidationError
from ai.validation import AssetAnalysisModel

# --- AssetAnalysisModel Tests ---

def test_asset_analysis_valid():
    data = {"ticker": "TEST", "adjustment_factor": 1.1, "confidence": 0.8, "reasoning": "Solid", "evidence": []}
    model = AssetAnalysisModel.model_validate(data)
    assert model.adjustment_factor == 1.1
    assert model.confidence == 0.8

@pytest.mark.parametrize("invalid_data", [
    {"ticker": "TEST", "adjustment_factor": 5.0, "confidence": 0.8, "reasoning": "Factor too high", "evidence": []},
    {"ticker": "TEST", "adjustment_factor": 1.0, "confidence": -0.5, "reasoning": "Confidence too low", "evidence": []},
    {"ticker": "TEST", "confidence": 0.8, "reasoning": "Missing factor", "evidence": []},
    {"ticker": "TEST", "adjustment_factor": 1.1, "confidence": "high", "reasoning": "Wrong type", "evidence": []},
])
def test_asset_analysis_invalid(invalid_data):
    with pytest.raises(ValidationError):
        AssetAnalysisModel.model_validate(invalid_data)

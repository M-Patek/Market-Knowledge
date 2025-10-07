# tests/test_ai_validation.py

import pytest
from pydantic import ValidationError
from ai.validation import AssetAnalysisModel, MarketSentimentModel

# --- AssetAnalysisModel Tests ---

def test_asset_analysis_valid():
    data = {"adjustment_factor": 1.1, "confidence": 0.8, "reasoning": "Solid"}
    model = AssetAnalysisModel.model_validate(data)
    assert model.adjustment_factor == 1.1
    assert model.confidence == 0.8

@pytest.mark.parametrize("invalid_data", [
    {"adjustment_factor": 5.0, "confidence": 0.8, "reasoning": "Factor too high"},
    {"adjustment_factor": 1.0, "confidence": -0.5, "reasoning": "Confidence too low"},
    {"confidence": 0.8, "reasoning": "Missing factor"},
    {"adjustment_factor": 1.1, "confidence": "high", "reasoning": "Wrong type"},
])
def test_asset_analysis_invalid(invalid_data):
    with pytest.raises(ValidationError):
        AssetAnalysisModel.model_validate(invalid_data)


# --- MarketSentimentModel Tests ---

def test_market_sentiment_valid():
    data = {"sentiment_score": -0.5, "reasoning": "Bearish"}
    model = MarketSentimentModel.model_validate(data)
    assert model.sentiment_score == -0.5

@pytest.mark.parametrize("invalid_data", [
    {"sentiment_score": 1.1, "reasoning": "Score too high"},
    {"sentiment_score": -1.5, "reasoning": "Score too low"},
    {"reasoning": "Missing score"},
    {"sentiment_score": "positive", "reasoning": "Wrong type"},
])
def test_market_sentiment_invalid(invalid_data):
    with pytest.raises(ValidationError):
        MarketSentimentModel.model_validate(invalid_data)

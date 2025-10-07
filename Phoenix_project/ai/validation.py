# ai/validation.py

from pydantic import BaseModel, Field


class AssetAnalysisModel(BaseModel):
    """Pydantic model for validating a single asset's analysis."""
    adjustment_factor: float = Field(..., ge=0.3, le=2.0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str


class MarketSentimentModel(BaseModel):
    """Pydantic model for validating market sentiment analysis."""
    sentiment_score: float = Field(..., ge=-1.0, le=1.0)
    reasoning: str

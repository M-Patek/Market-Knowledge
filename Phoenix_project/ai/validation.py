# ai/validation.py

from pydantic import BaseModel, Field


class AssetAnalysisModel(BaseModel):
    """Pydantic model for validating a single asset's analysis."""
    adjustment_factor: float = Field(..., ge=0.5, le=1.5)
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str


class MarketSentimentModel(BaseModel):
    """Pydantic model for validating market sentiment analysis."""
    sentiment_score: float = Field(..., ge=-1.0, le=1.0, description="Overall market sentiment score from -1.0 (bearish) to +1.0 (bullish)")
    reasoning: str

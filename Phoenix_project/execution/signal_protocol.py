from pydantic import BaseModel, Field
from typing import Literal

class StrategySignal(BaseModel):
    """
    (L6 Task 1) A standardized, Pydantic-based data protocol for communicating
    a strategic decision from the CognitiveEngine to the OrderManager.
    This enforces a strict contract between the strategy and execution layers.
    """
    ticker: str = Field(
        ...,
        description="The stock ticker symbol for the signal."
    )
    
    action: Literal["BUY", "SELL", "HOLD"] = Field(
        ...,
        description="The desired trading action."
    )
    
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="The confidence level of the signal, from 0.0 to 1.0."
    )
    
    target_weight: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="The target portfolio weight for the asset, from 0.0 (fully divested) to 1.0 (max allocation)."
    )

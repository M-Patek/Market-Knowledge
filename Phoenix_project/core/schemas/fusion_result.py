from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, model_validator, ConfigDict
from schemas.data_schema import MarketEvent, AnalystOpinion

class L1AgentResult(BaseModel):
    model_config = ConfigDict(extra='ignore')
    agent_name: str
    ticker: str
    output: Dict[str, Any]

class FusionResult(BaseModel):
    """
    Represents the output of the L2 Fusion Engine.
    This is the final, consolidated output after L2 Bayesian Fusion,
    Contradiction Detection, and Metacognitive analysis.
    """
    model_config = ConfigDict(extra='ignore')
    ticker: str
    final_sentiment: str  # "Bullish", "Bearish", "Neutral"
    confidence_score: float  # 0.0 to 1.0
    uncertainty_score: float # 0.0 to 1.0
    sentiment_reasoning: str
    
    # Raw inputs that led to this fusion
    l1_agent_results: List[L1AgentResult]
    
    # Structured data extracted and verified
    contributing_events: List[MarketEvent]
    contributing_opinions: List[AnalystOpinion]
    
    # List of detected contradictions
    contradictions_found: List[str]
    
    # Optional field for the reasoning trace from the metacognitive agent
    meta_reasoning_trace: Optional[str] = None

    @model_validator(mode='after')
    def check_sentiment_and_scores(self):
        sentiment = self.final_sentiment
        conf_score = self.confidence_score
        uncert_score = self.uncertainty_score

        if sentiment not in ["Bullish", "Bearish", "Neutral"]:
            raise ValueError("Final sentiment must be Bullish, Bearish, or Neutral")
        if not (0.0 <= conf_score <= 1.0):
            raise ValueError("Confidence score must be between 0.0 and 1.0")
        if not (0.0 <= uncert_score <= 1.0):
            raise ValueError("Uncertainty score must be between 0.0 and 1.0")
        return self

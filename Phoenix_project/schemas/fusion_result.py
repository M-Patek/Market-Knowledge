from pydantic import BaseModel, Field
from typing import List, Tuple, Dict, Any

class FusionResult(BaseModel):
    """
    (L3) Standardized data structure for the output of the BayesianFusionEngine.
    """
    posterior: Dict[str, float] = Field(
        ...,
        description="The resulting posterior probabilities for different outcomes (e.g., {'bullish': 0.7, 'bearish': 0.3})."
    )
    
    confidence_interval: Tuple[float, float] = Field(
        ...,
        description="The confidence interval for the primary posterior estimate."
    )
    
    rationale: str = Field(
        ...,
        description="A natural language explanation of how the fusion result was reached."
    )
    
    conflict_log: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="A log of any detected contradictions between evidence items."
    )

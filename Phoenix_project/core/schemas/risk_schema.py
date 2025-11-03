"""
Defines the Pydantic schema for the output of the L3 RiskAgent.
"""
from pydantic import BaseModel, Field
from typing import Dict, Any
import uuid

class RiskAdjustment(BaseModel):
    """
    A structured capital adjustment decision from the L3 RiskAgent.
    This is used by the execution layer to modify trade size.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for the risk adjustment.")
    agent_id: str = Field(..., description="The ID of the RiskAgent that generated this.")
    target_symbol: str = Field(..., description="The asset symbol this adjustment pertains to.")
    
    capital_modifier: float = Field(..., description="The capital allocation ratio (e.g., 0.0 to 1.0). 1.0 = full allocation.", ge=0.0, le=1.0)
    reasoning: str = Field(..., description="The reasoning for the adjustment (e.g., 'High uncertainty', 'Low volatility').")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Other metadata, e.g., uncertainty score, VIX level.")

    class Config:
        frozen = True

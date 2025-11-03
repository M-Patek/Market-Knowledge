"""
Defines the Pydantic schema for the output of the L2 AdversaryAgent.
"""
from pydantic import BaseModel, Field
from typing import Dict, Any
import uuid

class AdversaryResult(BaseModel):
    """
    A structured counter-argument or "pressure test" against a
    single L1 EvidenceItem. This is the standardized output
    of the AdversaryAgent.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for the counter-argument.")
    agent_id: str = Field(..., description="The ID of the AdversaryAgent that generated this.")
    target_evidence_id: str = Field(..., description="The 'id' of the L1 EvidenceItem that was tested.")
    
    counter_argument: str = Field(..., description="The textual counter-argument or alternative scenario.")
    is_challenge_successful: bool = Field(..., description="Whether the counter-argument successfully invalidates or weakens the original evidence.")
    confidence_impact: float = Field(..., description="A suggested impact factor on the original evidence's confidence (e.g., -0.2 for a 20% reduction).")
    
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Other metadata, e.g., prompt used, alternative scenario tested.")

    class Config:
        frozen = True

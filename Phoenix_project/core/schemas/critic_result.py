"""
Defines the Pydantic schema for the output of the L2 CriticAgent.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import uuid

class CriticResult(BaseModel):
    """
    A structured critique of a single L1 EvidenceItem.
    This is the standardized output of the CriticAgent.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for the critique.")
    agent_id: str = Field(..., description="The ID of the CriticAgent that generated this critique.")
    target_evidence_id: str = Field(..., description="The 'id' of the L1 EvidenceItem that was critiqued.")
    
    is_valid: bool = Field(..., description="Whether the evidence passed the critique (True/False).")
    critique: str = Field(..., description="The textual critique, highlighting flaws, biases, or supporting facts.")
    confidence_adjustment: float = Field(..., description="A suggested adjustment factor for the evidence's confidence (e.g., 0.8 for 20% reduction).")
    
    flags: List[str] = Field(default_factory=list, description="List of detected issues (e.g., 'BIAS', 'DATA_MISSING', 'LOGIC_FLAW').")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Other metadata, e.g., prompt used.")

    class Config:
        frozen = True

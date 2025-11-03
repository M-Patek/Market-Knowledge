"""
Defines the Pydantic schema for the output of the L2 MetacognitiveAgent.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import uuid

class SupervisionResult(BaseModel):
    """
    A structured analysis of an agent's Chain of Thought (CoT) or
    the overall pipeline's reasoning. This is the standardized
    output of the MetacognitiveAgent.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for the supervision log.")
    agent_id: str = Field(..., description="The ID of the MetacognitiveAgent that generated this.")
    
    analysis_summary: str = Field(..., description="The textual summary of the supervision (e.g., 'Detected hallucination in L1-Technial').")
    target_agent_ids: List[str] = Field(default_factory=list, description="The agent(s) being monitored or analyzed.")
    flags: List[str] = Field(default_factory=list, description="List of detected issues (e.g., 'HALLUCINATION', 'DIVERGENCE', 'GROUPTHINK').")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Other metadata, e.g., prompt used, CoT traces reviewed.")

    class Config:
        frozen = True

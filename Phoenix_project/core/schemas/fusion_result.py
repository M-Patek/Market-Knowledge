"""
Pydantic schemas for AI agent outputs (AgentDecision) and the final
combined result (FusionResult).
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

# --- 新增的 AgentDecision Schema ---
class AgentDecision(BaseModel):
    """
    Standardized output schema for a single AI agent's decision.
    Each agent in the ensemble must return this structure.
    """
    agent_name: str = Field(..., description="Name of the agent that produced this decision")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of when the decision was made")
    
    # Core Decision Components
    confidence: float = Field(..., description="Agent's confidence in its assessment (0.0 to 1.0)", ge=0, le=1)
    sentiment: float = Field(..., description="Agent's sentiment assessment (-1.0 to 1.0)", ge=-1, le=1)
    predicted_impact: float = Field(..., description="Agent's predicted impact score (-1.0 to 1.0)", ge=-1, le=1)
    
    # Rationale
    rationale: str = Field(..., description="A brief, clear justification for the decision")
    key_evidence_ids: List[str] = Field(default_factory=list, description="List of event_ids or doc_ids from the context that were most influential")
    
    # Optional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Any other relevant data (e.g., predicted timeframe, specific metrics)")

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

# --- 新增的 FusionResult Schema ---
class FusionResult(BaseModel):
    """
    Schema for the final, synthesized result after fusing multiple
    AgentDecisions. This is the ultimate output of the cognitive layer.
    """
    fusion_timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of when the fusion was completed")
    
    # Fused (Final) Decision
    fused_confidence: float = Field(..., description="Combined confidence score (0.0 to 1.0)", ge=0, le=1)
    fused_sentiment: float = Field(..., description="Combined sentiment score (-1.0 to 1.0)", ge=-1, le=1)
    fused_predicted_impact: float = Field(..., description="Combined impact score (-1.0 to 1.0)", ge=-1, le=1)
    
    # Uncertainty Metric
    cognitive_uncertainty: float = Field(..., description="Calculated uncertainty or disagreement among agents (0.0 to 1.0)", ge=0, le=1)
    
    # Rationale and Lineage
    fused_rationale: str = Field(..., description="Synthesized rationale explaining the final decision and how agent opinions were weighed")
    contributing_decisions: List[AgentDecision] = Field(..., description="The list of individual AgentDecisions that were used in this fusion")
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

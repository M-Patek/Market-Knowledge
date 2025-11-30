"""
Defines the L2 output schema: FusionResult.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum
from datetime import datetime, timezone
import uuid

# [Phase III Fix] System Status Enum for Zero Trap Prevention
class SystemStatus(str, Enum):
    OK = "OK"
    DEGRADED = "DEGRADED"
    HALT = "HALT"

class FusionResult(BaseModel):
    """
    The unified, high-confidence preliminary decision from the L2 Fusion Agent.
    This is the primary input for the L3 AlphaAgent.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for the fusion result.")
    # [Phase I Fix] Use timezone-aware UTC
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp (UTC) of when the decision was fused.")
    
    target_symbol: str = Field(..., description="The asset symbol this decision pertains to (e.g., 'AAPL').")
    
    decision: str = Field(..., description="The final, unified decision (e.g., 'STRONG_BUY', 'SELL', 'HOLD', 'NEUTRAL').")
    confidence: float = Field(..., description="The overall confidence in this unified decision (0.0 to 1.0).", ge=0.0, le=1.0)
    sentiment_score: float = Field(default=0.0, description="Normalized sentiment score (-1.0 to 1.0) for numerical consumption by L3.", ge=-1.0, le=1.0)
    
    reasoning: str = Field(..., description="A summary of the reasoning, including how L1 evidence was synthesized and conflicts resolved.")
    uncertainty: float = Field(..., description="A quantified score of the overall uncertainty in the decision.", ge=0.0, le=1.0)
    
    # [Phase I/III Fix] Fail-Closed Default: Default to HALT to prevent executing on silent failures.
    system_status: SystemStatus = Field(default=SystemStatus.HALT, description="Operational status of the inference engine (OK, DEGRADED, HALT).")
    
    supporting_evidence_ids: List[str] = Field(default_factory=list, description="List of IDs from the L1 EvidenceItems that support this decision.")
    conflicting_evidence_ids: List[str] = Field(default_factory=list, description="List of IDs from L1 EvidenceItems that conflict with this decision.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Other metadata, e.g., uncertainty dimensions, source L2 criticism IDs.")

    class Config:
        frozen = False
        # Optional: Add use_enum_values = True if needed for simple serialization

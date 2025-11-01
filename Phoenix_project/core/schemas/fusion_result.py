from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

# 修复：
# 从：from ..core.schemas.data_schema import MarketData, NewsData (路径错误)
# 改为：from .data_schema import MarketData, NewsData (同目录相对导入)
from .data_schema import MarketData, NewsData

class FusionResult(BaseModel):
    """
    Schema for the final L3 synthesized output.
    This represents the "final answer" of the cognitive pipeline.
    """
    query_id: str = Field(..., description="Unique ID for the original query")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Core Insight
    insight: str = Field(..., description="The synthesized, human-readable insight.")
    confidence: float = Field(..., description="Confidence score (0.0 to 1.0) for the insight.")
    
    # Supporting Evidence
    supporting_evidence: List[Dict[str, Any]] = Field(default_factory=list, description="List of key evidence snippets used.")
    conflicting_evidence: List[Dict[str, Any]] = Field(default_factory=list, description="List of evidence that conflicts.")
    
    # Actionable Decision (if applicable)
    suggested_action: Optional[str] = Field(None, description="e.g., 'BUY', 'SELL', 'HOLD', 'MONITOR'")
    action_parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters for the action (e.g., {'symbol': 'TSLA', 'target_price': 300})")
    
    # Context
    market_context: Optional[MarketData] = Field(None, description="Snapshot of market data at decision time.")
    news_context: List[NewsData] = Field(default_factory=list, description="Key news articles influencing the decision.")
    
    # Metadata
    reasoning_trace: List[str] = Field(default_factory=list, description="High-level trace of the reasoning steps.")
    contributing_agents: List[str] = Field(default_factory=list, description="Agents that contributed to this insight.")
    
    class Config:
        arbitrary_types_allowed = True
        validate_assignment = True # Ensures type checks on assignment


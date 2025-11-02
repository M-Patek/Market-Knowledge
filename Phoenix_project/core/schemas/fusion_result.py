"""
定义认知引擎的输出：AgentDecision 和 FusionResult。
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

# FIX (E3): 添加缺失的 AgentDecision schema
class AgentDecision(BaseModel):
    """
    单个AI智能体的决策输出。
    """
    agent_name: str = Field(..., description="智能体名称 (e.g., 'technical_analyst')")
    timestamp: datetime = Field(..., description="决策生成时间 (UTC)")
    decision: str = Field(..., description="决策 (e.g., 'BULLISH', 'BEARISH', 'NEUTRAL')")
    confidence: float = Field(..., description="置信度 (0.0 to 1.0)", ge=0.0, le=1.0)
    reasoning: str = Field(..., description="决策的详细推理过程 (Chain of Thought)")
    supporting_evidence: List[Dict[str, Any]] = Field(default_factory=list, description="支持该决策的数据点或引用")
    error_analysis: Optional[str] = Field(None, description="智能体对其潜在错误的分析")

class FusionResult(BaseModel):
    """
    认知引擎在分析一组 AgentDecision 后得出的最终融合结果。
    这是认知层的主要输出，用于驱动投资组合构建。
    """
    id: str = Field(..., description="融合结果的唯一ID")
    timestamp: datetime = Field(..., description="融合决策生成时间 (UTC)")
    
    # FIX (E3): 确保 FusionResult 包含 AgentDecision 列表
    agent_decisions: List[AgentDecision] = Field(..., description="参与此次融合的所有智能体的决策")
    
    final_decision: str = Field(..., description="最终的综合决策 (e.g., 'STRONG_BUY', 'SELL', 'HOLD')")
    final_confidence: float = Field(..., description="最终决策的综合置信度", ge=0.0, le=1.0)
    
    summary: str = Field(..., description="对所有推理的总结")
    conflicts_identified: List[str] = Field(default_factory=list, description="识别出的智能体间的主要分歧点")
    conflict_resolution: Optional[str] = Field(None, description="分歧的解决方案或仲裁结果")
    
    uncertainty_score: float = Field(..., description="量化的不确定性得分", ge=0.0, le=1.0)
    uncertainty_dimensions: Dict[str, float] = Field(default_factory=dict, description="不确定性的分解 (e.g., 'data_gap', 'model_disagreement')")
    
    metadata: Dict[str, Any] = Field(default_factory=dict, description="与此融合相关的其他元数据 (e.g., 'target_symbol')")

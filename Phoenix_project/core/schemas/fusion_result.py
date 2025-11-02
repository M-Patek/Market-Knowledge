"""
FusionResult Schema
- 定义认知引擎中 "Fusion" (融合) 步骤的输出数据结构。
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Any

class FusionResult(BaseModel):
    """
    保存来自 MetacognitiveAgent 融合步骤的结构化输出。
    """
    
    # (假设的现有字段)
    insights: List[Any] = Field(default_factory=list, description="融合后的见解列表")
    confidence_scores: List[float] = Field(default_factory=list, description="每个见解的置信度")
    
    # 关键修正 (Error 8):
    # 添加 UncertaintyGuard (不确定性守卫) 所需的字段
    
    status: str = Field(default="PENDING", description="融合结果的状态 (e.g., PENDING, SUCCESS, FAILED)")
    
    final_decision: Optional[Any] = Field(None, description="融合后的最终决策或信号")
    
    error_message: Optional[str] = Field(None, description="如果 status 为 FAILED，记录错误信息")


    class Config:
        """
        Pydantic 配置
        """
        # (允许模型中存在任意类型，例如复杂的分析对象)
        arbitrary_types_allowed = True
        validate_assignment = True

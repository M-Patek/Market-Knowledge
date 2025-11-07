"""
Defines the Pydantic schema for the output of the L3 RiskAgent.

[主人喵的修复]
添加 RiskReport schema，供 RiskManager.evaluate_and_adjust 使用。
"""
from pydantic import BaseModel, Field
from typing import Dict, Any, List
from datetime import datetime # [主人喵的修复] 导入 datetime
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

class RiskReport(BaseModel):
    """
    [主人喵的修复]
    由 RiskManager.evaluate_and_adjust 生成的报告。
    记录所有采取的风险调整措施。
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    adjustments_made: List[str] = Field(default_factory=list, description="记录风险调整的字符串列表")
    portfolio_risk_metrics: Dict[str, Any] = Field(default_factory=dict, description="计算出的投资组合风险指标 (e.G., VaR, 集中度)")
    passed: bool = Field(True, description="投资组合是否通过所有风险检查")

"""
Defines the Pydantic schema for the output of the L3 RiskAgent.

[主人喵的修复]
添加 RiskReport schema，供 RiskManager.evaluate_and_adjust 使用。
[Phase 0] 添加 SignalType 和 RiskSignal 类层次结构。
"""
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from enum import Enum
from datetime import datetime
import uuid

class SignalType(str, Enum):
    MARKET_RISK = "market_risk"
    CIRCUIT_BREAKER = "circuit_breaker"
    LIQUIDITY_RISK = "liquidity_risk"
    OPERATIONAL_RISK = "operational_risk"

class RiskParameter(BaseModel):
    name: str
    value: Any
    threshold: Optional[Any] = None

class RiskSignal(BaseModel):
    """Base class for risk signals detected by the Risk Manager."""
    type: SignalType = Field(default=SignalType.MARKET_RISK)
    description: str
    triggers_circuit_breaker: bool = Field(default=False)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class DrawdownSignal(RiskSignal):
    """Signal for maximum drawdown violations."""
    current_drawdown: float
    max_drawdown: float

class ConcentrationSignal(RiskSignal):
    """Signal for position concentration violations."""
    symbol: str
    current_concentration: float
    max_concentration: float

class VolatilitySignal(RiskSignal):
    """Signal for excessive volatility."""
    symbol: str
    current_volatility: float
    volatility_threshold: float

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

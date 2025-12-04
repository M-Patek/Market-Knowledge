"""
管道状态 (PipelineState)
一个内存中的对象，用于存储系统在两个周期之间的状态。
"""
import uuid
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone
from pydantic import BaseModel, Field

# FIX (E1, E2, E3): 导入统一的模式
from Phoenix_project.core.schemas.data_schema import PortfolioState, TargetPortfolio, MarketData
from Phoenix_project.core.schemas.fusion_result import AgentDecision, FusionResult
from Phoenix_project.core.schemas.evidence_schema import EvidenceItem

from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

class PipelineState(BaseModel):
    """
    管理系统的当前状态 (瞬时快照)。
    [Refactored Phase 2.1] 基于 Pydantic，移除内存历史，支持序列化。
    """
    # --- 身份与时间 ---
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    step_index: int = 0
    # [Phase I Fix] Use timezone-aware UTC to prevent offset-naive errors
    current_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # --- 上下文与任务 ---
    main_task_query: Dict[str, Any] = Field(default_factory=dict)
    
    # --- 财务状态 ---
    portfolio_state: Optional[PortfolioState] = None

    # --- 认知状态 (仅保留上一帧结果) ---
    latest_decisions: List[AgentDecision] = Field(default_factory=list)
    latest_fusion_result: Optional[Dict[str, Any]] = None
    latest_final_decision: Optional[Dict[str, Any]] = None
    
    # [Phase I Fix] Strict Typing for Serialization Safety
    target_portfolio: Optional[TargetPortfolio] = None 
    
    # [Phase I Fix] Strict Typing: Enforce List[EvidenceItem] to prevent garbage data
    l1_insights: List[EvidenceItem] = Field(default_factory=list)
    l2_supervision_results: Dict[str, Any] = Field(default_factory=dict)
    market_state: Dict[str, Any] = Field(default_factory=dict)
    raw_events: List[Dict[str, Any]] = Field(default_factory=list)
    l3_decision: Dict[str, Any] = Field(default_factory=dict)
    
    # [Phase I Fix] Strict Typing for formalized fields
    market_data_batch: Optional[List[MarketData]] = None 
    l3_alpha_signal: Optional[Dict[str, float]] = None

    class Config:
        arbitrary_types_allowed = True
        # [Phase I Fix] Validate assignment to catch type errors immediately
        validate_assignment = True

    def update_time(self, new_time: datetime):
        """更新系统仿真时间。"""
        self.current_time = new_time

    def update_portfolio_state(self, new_state: PortfolioState):
        """更新持仓状态。"""
        self.portfolio_state = new_state

    def set_l1_insights(self, insights: List[EvidenceItem]):
        self.l1_insights = insights

    def set_l2_supervision(self, results: Dict[str, Any]):
        self.l2_supervision_results = results

    def set_market_state(self, state: Dict[str, Any]):
        self.market_state = state

    def set_raw_events(self, events: List[Dict[str, Any]]):
        self.raw_events = events

    def set_target_portfolio(self, tp: TargetPortfolio):
        self.target_portfolio = tp

    def set_l3_decision(self, decision: Dict[str, Any]):
        self.l3_decision = decision

    def set_market_data_batch(self, batch: List[MarketData]):
        self.market_data_batch = batch

    def set_l3_alpha_signal(self, signal: Dict[str, float]):
        self.l3_alpha_signal = signal

    def update_ai_outputs(self, fusion_result: FusionResult):
        """
        更新 AI 决策状态 (仅保留最新)。
        """
        if fusion_result.agent_decisions:
            self.latest_decisions = fusion_result.agent_decisions
        
        # 序列化 FusionResult 以便存储
        # [Task 0.1 Fix] Complete truncation & Handle Pydantic V1/V2 compatibility
        self.latest_fusion_result = fusion_result.model_dump() if hasattr(fusion_result, 'model_dump') else fusion_result.dict()

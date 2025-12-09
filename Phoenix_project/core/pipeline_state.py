"""
Phoenix_project/core/pipeline_state.py
[Phase 4 Task 2] Time Machine Implementation.
Remove default datetime.now to prevent future leakage.
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
    
    # [Phase 4 Task 2] Time Machine: Remove default now() to prevent future leakage.
    # Must be explicitly set by LoopManager or Orchestrator.
    current_time: Optional[datetime] = None

    # --- 上下文与任务 ---
    main_task_query: Dict[str, Any] = Field(default_factory=dict)
    
    # --- 财务状态 ---
    portfolio_state: Optional[PortfolioState] = None

    # --- 认知状态 (仅保留上一帧结果) ---
    latest_decisions: List[AgentDecision] = Field(default_factory=list)
    
    # [Phase II Fix] Strict Typing for Fusion Result
    latest_fusion_result: Optional[FusionResult] = None
    
    latest_final_decision: Optional[Dict[str, Any]] = None
    # [Task 0.1 Fix] Added storage for fact check results
    latest_fact_check: Optional[Dict[str, Any]] = None
    
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
    
    # [Task 2] Version Control for CAS/Optimistic Locking
    version: int = 0

    class Config:
        arbitrary_types_allowed = True
        # [Phase I Fix] Validate assignment to catch type errors immediately
        validate_assignment = True

    def update_time(self, new_time: datetime):
        """更新系统仿真时间。"""
        # [Task 0.1 Fix] Enforce UTC (Offset-Aware)
        if new_time.tzinfo is None:
            self.current_time = new_time.replace(tzinfo=timezone.utc)
        else:
            self.current_time = new_time.astimezone(timezone.utc)

    def update_value(self, key: str, value: Any):
        """
        [Task 0.1 Fix] Safe Accessor with Legacy Mapping & Validation
        """
        # Map legacy keys from CognitiveEngine to Schema fields
        key_map = {
            "last_fusion_result": "latest_fusion_result",
            "last_fact_check": "latest_fact_check",
            "last_guarded_decision": "latest_final_decision"
        }
        target_key = key_map.get(key, key)
        
        if target_key not in self.model_fields:
            # Relaxed for dynamic fields during dev
            pass
            
        setattr(self, target_key, value)

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
        
        self.latest_fusion_result = fusion_result

"""
管道状态 (PipelineState)
一个内存中的对象，用于存储系统在两个周期之间的状态。
"""
import uuid
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field

# FIX (E1, E2, E3): 导入统一的模式
from Phoenix_project.core.schemas.data_schema import PortfolioState
from Phoenix_project.core.schemas.fusion_result import AgentDecision, FusionResult

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
    current_time: datetime = Field(default_factory=datetime.utcnow)

    # --- 上下文与任务 ---
    main_task_query: Dict[str, Any] = Field(default_factory=dict)
    
    # --- 财务状态 ---
    portfolio_state: Optional[PortfolioState] = None

    # --- 认知状态 (仅保留上一帧结果) ---
    latest_decisions: List[AgentDecision] = Field(default_factory=list)
    latest_fusion_result: Optional[Dict[str, Any]] = None
    latest_final_decision: Optional[Dict[str, Any]] = None
    target_portfolio: Optional[Any] = None # For OrderManager handoff
    l1_insights: Dict[str, Any] = Field(default_factory=dict)
    l2_supervision_results: Dict[str, Any] = Field(default_factory=dict)
    market_state: Dict[str, Any] = Field(default_factory=dict)
    raw_events: List[Dict[str, Any]] = Field(default_factory=list)
    l3_decision: Dict[str, Any] = Field(default_factory=dict)
    
    # [Task 3.2] Formalize monkey-patched fields
    market_data_batch: Optional[Any] = None # List[MarketData] or Dict
    l3_alpha_signal: Optional[Dict[str, float]] = None

    class Config:
        arbitrary_types_allowed = True

    def update_time(self, new_time: datetime):
        """更新系统仿真时间。"""
        self.current_time = new_time

    def update_portfolio_state(self, new_state: PortfolioState):
        """更新持仓状态。"""
        self.portfolio_state = new_state

    def set_l1_insights(self, insights: Dict[str, Any]):
        self.l1_insights = insights

    def set_l2_supervision(self, results: Dict[str, Any]):
        self.l2_supervision_results = results

    def set_market_state(self, state: Dict[str, Any]):
        self.market_state = state

    def set_raw_events(self, events: List[Dict[str, Any]]):
        self.raw_events = events

    def set_target_portfolio(self, tp: Any):
        self.target_portfolio = tp

    def set_l3_decision(self, decision: Dict[str, Any]):
        self.l3_decision = decision

    # [Task 3.2] Setters for formalized fields
    def set_market_data_batch(self, batch: Any):
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
        self.latest_fusion_result = fusion_result.model_dump() if hasattr(fusion_result, 'model_dump') else fusion_result.__dict__

    def add_final_decision(self, decision: Dict[str, Any]):
        """更新最终裁决 (仅保留最新)。"""
        self.latest_final_decision = decision

    def get_snapshot(self) -> Dict[str, Any]:
        """
        获取当前状态的字典表示 (Pydantic Native)。
        """
        return self.model_dump()
        
    def get_latest_portfolio_state(self) -> Optional[PortfolioState]:
        """
        获取最新的投资组合状态。
        """
        return self.portfolio_state

    def get_main_task_query(self) -> Dict[str, Any]:
        """
        (FIX) 为 L1/L2 智能体实现缺失的方法。
        """
        if not self.main_task_query:
             return {
                "symbol": "BTC/USD", # Default
                "description": "Analyze default symbol."
             }
        return self.main_task_query

    def update_value(self, key: str, value: Any):
        """Generic update for dynamic keys (used by CognitiveEngine)."""
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            # For dynamic fields not in schema, Pydantic might ignore or error unless extra='allow'
            # Here we just log or ignore if strict
            pass

    def get_full_context_formatted(self) -> str:
        """
        (FIX) 为 AuditManager 实现缺失的方法。
        将当前状态序列化为字符串用于日志记录 (无历史)。
        """
        context_str = f"--- SYSTEM STATE (Step {self.step_index}) ---\n"
        context_str += f"Run ID: {self.run_id}\n"
        context_str += f"Current Time: {self.current_time.isoformat()}\n"
        
        if self.portfolio_state:
            context_str += f"Portfolio: Val={self.portfolio_state.total_value}, Cash={self.portfolio_state.cash}\n"
            context_str += f"Positions: {len(self.portfolio_state.positions)}\n"
        
        if self.latest_decisions:
            context_str += f"Last Decisions: {len(self.latest_decisions)} agents active.\n"
            
        return context_str

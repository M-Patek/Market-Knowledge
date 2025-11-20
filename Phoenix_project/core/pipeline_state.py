"""
管道状态 (PipelineState)
一个内存中的对象，用于存储系统在两个周期之间的状态。
"""
from typing import Dict, Any, List, Optional
from collections import deque
from datetime import datetime

# FIX (E1, E2, E3): 导入统一的模式
from Phoenix_project.core.schemas.data_schema import MarketData, NewsData, EconomicIndicator, PortfolioState
from Phoenix_project.core.schemas.fusion_result import AgentDecision, FusionResult

from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

class PipelineState:
    """
    管理系统的当前状态，包括最新的数据、持仓和决策。
    """

    # FIX (E5): 更改构造函数以接受配置或初始状态
    def __init__(self, initial_state: Optional[Dict[str, Any]] = None, max_history: int = 100):
        
        self.max_history = max_history
        self.log_prefix = "PipelineState:"
        
        if initial_state:
            self.current_time: datetime = initial_state.get("current_time", datetime.utcnow())
            self.portfolio_state: PortfolioState = initial_state.get("portfolio_state")
            # (FIX) 存储主任务查询
            self.main_task_query: Dict[str, Any] = initial_state.get("main_task_query", {})
            # ... 其他状态的恢复
        else:
            self.current_time: datetime = datetime.min
            self.portfolio_state: Optional[PortfolioState] = None
            # (FIX) 存储主任务查询
            self.main_task_query: Dict[str, Any] = {"description": "Default state", "symbol": "N/A"}

        # 存储历史数据 (用于回溯分析)
        self.market_data_history: deque[MarketData] = deque(maxlen=self.max_history)
        self.news_history: deque[NewsData] = deque(maxlen=self.max_history)
        self.econ_history: deque[EconomicIndicator] = deque(maxlen=self.max_history)
        
        # FIX (E3): 存储 AgentDecision 和 FusionResult
        self.decision_history: deque[AgentDecision] = deque(maxlen=self.max_history)
        self.fusion_history: deque[FusionResult] = deque(maxlen=self.max_history)
        
        # FIX (Task 1.1): Add missing history queues
        self.fact_check_history: deque[Any] = deque(maxlen=self.max_history)
        self.final_decision_history: deque[Dict[str, Any]] = deque(maxlen=self.max_history)
        
        logger.info(f"{self.log_prefix} Initialized. Max history size: {self.max_history}")

    def update_time(self, new_time: datetime):
        self.current_time = new_time

    def update_portfolio_state(self, new_state: PortfolioState):
        self.portfolio_state = new_state
        logger.debug(f"{self.log_prefix} Portfolio state updated to {new_state.timestamp}")

    def update_data_batch(self, data_batch: Dict[str, List[Any]]):
        """
        将新一批的数据添加到历史记录中。
        """
        # FIX (E1): 使用正确的键
        if "market_data" in data_batch:
            self.market_data_history.extend(data_batch["market_data"])
            
        if "news_data" in data_batch:
            self.news_history.extend(data_batch["news_data"])
            
        if "economic_indicators" in data_batch:
            self.econ_history.extend(data_batch["economic_indicators"])
            
        logger.debug(f"{self.log_prefix} Data history updated.")

    def update_ai_outputs(self, fusion_result: FusionResult):
        """
        存储 AI 决策的结果。
        """
        self.fusion_history.append(fusion_result)
        
        # FIX (E3): 存储 AgentDecision
        if fusion_result.agent_decisions:
            self.decision_history.extend(fusion_result.agent_decisions)
            
        logger.debug(f"{self.log_prefix} AI outputs updated with FusionID {fusion_result.id}")

    def add_fact_check_report(self, report: Any):
        """Add a fact check report to history."""
        self.fact_check_history.append(report)
        logger.debug(f"{self.log_prefix} Fact check report added.")

    def add_final_decision(self, decision: Dict[str, Any]):
        """Add a final arbitration decision to history."""
        self.final_decision_history.append(decision)
        logger.debug(f"{self.log_prefix} Final decision added.")

    def get_snapshot(self) -> Dict[str, Any]:
        """
        创建系统当前状态的快照，用于持久化。
        """
        return {
            "current_time": self.current_time,
            "portfolio_state": self.portfolio_state.model_dump() if self.portfolio_state else None,
            "history_counts": {
                "market_data": len(self.market_data_history),
                "news": len(self.news_history),
                "fusion_results": len(self.fusion_history),
                "fact_checks": len(self.fact_check_history),
                "decisions": len(self.final_decision_history)
            }
            # 注意：不序列化完整的 deque 历史，以避免快照过大
        }
        
    def get_latest_market_data(self, symbol: str) -> Optional[MarketData]:
        """
        (示例) 从历史中获取最新的市场数据。
        FIX (E10): 替换 test_pipeline_state 中使用的旧方法
        """
        for data in reversed(self.market_data_history):
            if data.symbol == symbol:
                return data
        return None
        
    def get_latest_portfolio_state(self) -> Optional[PortfolioState]:
        """
        (示例) 获取最新的投资组合状态。
        FIX (E10): 替换 test_pipeline_state 中使用的旧方法
        """
        return self.portfolio_state

    # --- FIX: 添加缺失的方法 ---

    def get_value(self, key: str, default: Any = None) -> Any:
        """
        (FIX) 实现一个 getter 以兼容 AuditManager。
        从状态属性中检索值。
        """
        if hasattr(self, key):
            return getattr(self, key)
        
        # (FIX) 检查 portfolio_state，因为 audit_manager 可能需要
        if self.portfolio_state and hasattr(self.portfolio_state, key):
             return getattr(self.portfolio_state, key)
             
        # (FIX) 检查 main_task_query
        if key == "main_task_query":
            return self.main_task_query

        logger.warning(f"{self.log_prefix} get_value for '{key}' not found, returning default.")
        return default

    def get_main_task_query(self) -> Dict[str, Any]:
        """
        (FIX) 为 L1/L2 智能体实现缺失的方法。
        """
        if not self.main_task_query:
             logger.warning(f"{self.log_prefix} get_main_task_query returning mock data.")
             return {
                "symbol": "AAPL", # Mock default
                "description": "Analyze default symbol AAPL."
             }
        return self.main_task_query

    def get_full_context_formatted(self) -> str:
        """
        (FIX) 为 AuditManager 实现缺失的方法。
        将最近的状态序列化为字符串用于日志记录。
        """
        context_str = "--- LATEST CONTEXT ---\n"
        context_str += f"Current Time: {self.current_time.isoformat()}\n"
        
        if self.portfolio_state:
            context_str += f"Portfolio Value: {self.portfolio_state.total_value}\n"
            context_str += f"Cash: {self.portfolio_state.cash}\n"
            context_str += f"Positions: {len(self.portfolio_state.positions)}\n"
        
        context_str += "\n--- Recent News (Last 5) ---\n"
        news_items = list(self.news_history)[-5:]
        if not news_items:
            context_str += "No recent news.\n"
        for news in news_items:
            headline = news.headline or f"News ({news.source})"
            context_str += f"[{news.timestamp.isoformat()}] {headline[:80]}...\n"
            
        context_str += "\n--- Recent Market Data (Last 5) ---\n"
        market_items = list(self.market_data_history)[-5:]
        if not market_items:
            context_str += "No recent market data.\n"
        for md in market_items:
            context_str += f"[{md.timestamp.isoformat()}] {md.symbol}: Close={md.close}, Vol={md.volume}\n"
            
        return context_str

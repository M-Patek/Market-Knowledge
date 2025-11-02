"""
管道状态 (PipelineState)
一个内存中的对象，用于存储系统在两个周期之间的状态。
"""
from typing import Dict, Any, List, Optional
from collections import deque
from datetime import datetime

# FIX (E1, E2, E3): 导入统一的模式
from core.schemas.data_schema import MarketData, NewsData, EconomicIndicator, PortfolioState
from core.schemas.fusion_result import AgentDecision, FusionResult

from monitor.logging import get_logger

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
            # ... 其他状态的恢复
        else:
            self.current_time: datetime = datetime.min
            self.portfolio_state: Optional[PortfolioState] = None

        # 存储历史数据 (用于回溯分析)
        self.market_data_history: deque[MarketData] = deque(maxlen=self.max_history)
        self.news_history: deque[NewsData] = deque(maxlen=self.max_history)
        self.econ_history: deque[EconomicIndicator] = deque(maxlen=self.max_history)
        
        # FIX (E3): 存储 AgentDecision 和 FusionResult
        self.decision_history: deque[AgentDecision] = deque(maxlen=self.max_history)
        self.fusion_history: deque[FusionResult] = deque(maxlen=self.max_history)
        
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
                "fusion_results": len(self.fusion_history)
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

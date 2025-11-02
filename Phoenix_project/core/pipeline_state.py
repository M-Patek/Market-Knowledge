import pandas as pd
from typing import Dict, Any, Optional, List
# 修正：[FIX-ImportError/TypeError] 'Deque' 应该从 'collections' 导入，而不是 'typing'
from collections import Deque 
import threading

class PipelineState:
    """
    一个线程安全的数据类，用于保存 Phoenix 系统的当前状态。
    
    这包括当前时间、市场状况、投资组合持仓、
    情绪指标以及最近事件的队列。
    
    它被设计为在所有组件之间共享。
    """

    def __init__(self, max_recent_events: int = 100):
        self.lock = threading.RLock() # 可重入锁，用于保护状态
        
        # --- 时间状态 ---
        self._current_time: pd.Timestamp = pd.Timestamp.utcnow()
        
        # --- 市场内部状态 (由 CognitiveEngine 更新) ---
        self._market_regime: str = "Unknown"
        self._volatility_index: float = 0.0
        self._sentiment_score: float = 0.0
        
        # --- 投资组合状态 (由 OrderManager/TLC 更新) ---
        self._portfolio_value: float = 0.0
        self._cash: float = 0.0
        self._positions: Dict[str, float] = {} # e.g., {"AAPL": 100.0}
        
        # --- 事件状态 ---
        self._recent_events: Deque[Dict[str, Any]] = Deque(maxlen=max_recent_events)
        
        # --- 系统健康状态 ---
        self._component_health: Dict[str, str] = {"Orchestrator": "OK"}
        self._last_error: Optional[str] = None

    def update_time(self, new_time: pd.Timestamp):
        with self.lock:
            self._current_time = new_time
            
    def get_current_time(self) -> pd.Timestamp:
        with self.lock:
            return self._current_time
            
    def update_market_state(self, regime: str, volatility: float, sentiment: float):
        with self.lock:
            self._market_regime = regime
            self._volatility_index = volatility
            self._sentiment_score = sentiment
            
    def get_market_state(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "regime": self._market_regime,
                "volatility": self._volatility_index,
                "sentiment": self._sentiment_score
            }

    def update_portfolio(self, total_value: float, cash: float, positions: Dict[str, float]):
        with self.lock:
            self._portfolio_value = total_value
            self._cash = cash
            self._positions = positions.copy() # 创建一个副本
            
    def get_portfolio_state(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "total_value": self._portfolio_value,
                "cash": self._cash,
                "positions": self._positions.copy()
            }
            
    def add_event(self, event: Dict[str, Any]):
        with self.lock:
            self._recent_events.appendleft(event) # 添加到队列前面
            
    def get_recent_events(self) -> List[Dict[str, Any]]:
        with self.lock:
            return list(self._recent_events)

    def update_component_health(self, component_name: str, status: str, error: Optional[str] = None):
        with self.lock:
            self._component_health[component_name] = status
            if error:
                self._last_error = f"[{component_name}] {error}"
                
    def get_system_health(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "components": self._component_health.copy(),
                "last_error": self._last_error
            }
            
    def get_full_state_snapshot(self) -> Dict[str, Any]:
        """
E        返回整个状态的一个深层复制快照。
        """
        with self.lock:
            return {
                "current_time": self._current_time,
                "market_state": self.get_market_state(),
                "portfolio_state": self.get_portfolio_state(),
                "system_health": self.get_system_health(),
                "recent_events_count": len(self._recent_events)
            }

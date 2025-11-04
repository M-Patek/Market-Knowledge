import pandas as pd
from typing import Dict, Any, Optional
from threading import RLock

# 修复：将相对导入 'from .core.pipeline_state...' 更改为绝对导入
from Phoenix_project.core.pipeline_state import PipelineState
# 修复：将相对导入 'from .monitor.logging...' 更改为绝对导入
from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

class ContextBus:
    """
    A thread-safe, in-memory singleton that holds the most recent,
    globally shared state of the system (the "context").
    
    This allows different components (e.g., Orchestrator, API Gateway,
    Risk Manager) to access the current market data, portfolio state, etc.,
    without direct coupling.
    """

    _instance = None
    _lock = RLock()

    def __new__(cls, *args, **kwargs):
        """Implements the singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ContextBus, cls).__new__(cls)
                    # Initialize custom attributes
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, strategy_name: str, initial_portfolio: Dict[str, float] = None):
        """
        Initializes the ContextBus singleton.
        
        Args:
            strategy_name (str): The name of the strategy.
            initial_portfolio (Dict, optional): The starting portfolio.
        """
        if self._initialized:
            return
            
        with self._lock:
            if self._initialized:
                return
                
            self.current_state = PipelineState(
                timestamp=pd.Timestamp.now(tz='UTC'),
                strategy_name=strategy_name,
                portfolio=initial_portfolio or {"CASH": 0.0},
                market_data={},
                recent_events=[],
                triggering_event=None
            )
            self._initialized = True
            logger.info("ContextBus singleton initialized.")

    def update_state(self, new_state: PipelineState):
        """
        Atomically updates the entire shared state.
        
        Args:
            new_state (PipelineState): The new state snapshot.
        """
        with self._lock:
            if new_state.timestamp < self.current_state.timestamp:
                logger.warning(f"Ignored out-of-order state update. "
                               f"Bus: {self.current_state.timestamp}, "
                               f"New: {new_state.timestamp}")
                return
                
            self.current_state = new_state
            logger.debug(f"ContextBus state updated to timestamp: {new_state.timestamp}")

    def get_current_state(self) -> PipelineState:
        """
        Returns a (shallow) copy of the current state.
        
        Returns:
            PipelineState: The most recent state.
        """
        with self._lock:
            # Return a copy to prevent mutation of the bus's internal state
            return self.current_state.model_copy()

    def get_latest_timestamp(self) -> pd.Timestamp:
        """Returns the timestamp of the current state."""
        with self._lock:
            return self.current_state.timestamp

    def get_current_portfolio(self) -> Dict[str, float]:
        """Returns a copy of the current portfolio."""
        with self._lock:
            return self.current_state.portfolio.copy()

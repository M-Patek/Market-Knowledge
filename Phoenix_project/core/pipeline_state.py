import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from collections import deque
from copy import deepcopy
from core.schemas.data_schema import MarketData, NewsData, EconomicIndicator
from core.schemas.fusion_result import FusionResult, AgentDecision
from monitor.logging import get_logger

logger = get_logger(__name__)

class PipelineState:
    """
    A thread-safe, asynchronous state manager for the application.
    It holds all data, decisions, and portfolio information for
    a single processing cycle.
    
    Uses asyncio.Lock to ensure safe concurrent access, although
    in the current design, it's mostly accessed serially within
    the main cycle. The lock is good practice.
    """

    def __init__(self, initial_state: Optional[Dict[str, Any]] = None):
        self._state: Dict[str, Any] = {
            # --- Core Timestamps ---
            "current_time": datetime.utcnow(),
            "cycle_start_time": None,
            "last_successful_cycle_time": None,
            "last_data_ingest_time": None,
            
            # --- Cycle-specific IDs ---
            "current_decision_id": None,
            
            # --- Raw Data Buffers (Time-Windowed) ---
            "market_data": deque(maxlen=200), # Store last N ticks
            "news_data": deque(maxlen=50),   # Store last N articles
            "economic_data": deque(maxlen=50),
            
            # --- Cognitive Artifacts (from last cycle) ---
            "last_formatted_context": None,
            "last_agent_decisions": [], # List[AgentDecision]
            "last_arbitration": None,
            "last_fusion_result": None, # FusionResult
            "last_fact_check": None,
            "last_guarded_decision": None, # FusionResult
            
            # --- Execution & Portfolio State ---
            "portfolio": {
                "cash": 100000.0,
                "total_value": 100000.0,
                "positions": {} # e.g., {"AAPL": {"size": 10, "avg_price": 150.0, "market_value": ...}}
            },
            "open_orders": [], # List[Order]
            "trade_history": deque(maxlen=1000), # List[Execution]
            
            # --- System Health ---
            "last_cycle_time_ms": 0.0,
            "system_status": "RUNNING", # RUNNING, DEGRADED, CIRCUIT_BREAKER
            
            **(initial_state or {})
        }
        self._lock = asyncio.Lock()
        logger.info("PipelineState initialized.")

    async def get_value(self, key: str, default: Any = None) -> Any:
        """Safely get a value from the state."""
        async with self._lock:
            return deepcopy(self._state.get(key, default))

    async def update_value(self, key: str, value: Any):
        """Safely update a single key in the state."""
        async with self._lock:
            self._state[key] = value
            # logger.debug(f"State updated: {key} = {value}")

    async def update_state(self, updates: Dict[str, Any]):
        """Safely update multiple keys in the state."""
        async with self._lock:
            for key, value in updates.items():
                # Handle special deque appends
                if key in ("market_data", "news_data", "economic_data") and isinstance(value, list):
                    self._state[key].extend(value)
                elif key in ("market_data", "news_data", "economic_data") and not isinstance(value, (list, deque)):
                     self._state[key].append(value)
                elif key == "trade_history" and isinstance(value, list):
                    self._state[key].extend(value)
                elif key == "trade_history" and not isinstance(value, (list, deque)):
                    self._state[key].append(value)
                else:
                    self._state[key] = value
            # logger.debug(f"Batch state update applied: {updates.keys()}")

    async def get_full_state_copy(self) -> Dict[str, Any]:
        """Return a deep copy of the entire state dictionary."""
        async with self._lock:
            return deepcopy(self._state)

    # --- Convenience Accessors for Context Building ---

    async def get_full_context(self) -> Dict[str, List[Any]]:
        """
        Get all data needed to build a context snapshot for the AI.
        Returns copies to prevent mutation.
        """
        async with self._lock:
            return {
                "market_data": list(self._state["market_data"]),
                "news_data": list(self._state["news_data"]),
                "economic_data": list(self._state["economic_data"]),
                # Add other context-relevant data here
            }

    async def get_full_context_formatted(self) -> str:
        """
        Get the last formatted context string.
        Assumes DataAdapter has run and stored this.
        """
        async with self._lock:
            return self._state.get("last_formatted_context", "No context available.")

    async def get_latest_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get the most recent market data for a specific symbol."""
        async with self._lock:
            for data in reversed(self._state["market_data"]):
                if data.symbol == symbol:
                    return deepcopy(data)
            return None
    
    async def get_portfolio(self) -> Dict[str, Any]:
        """Safely get a copy of the portfolio."""
        async with self._lock:
            return deepcopy(self._state["portfolio"])

    async def update_portfolio(self, portfolio_update: Dict[str, Any]):
        """
        Safely update the portfolio state.
        This is typically called by the OrderManager or a portfolio
        update service.
        """
        async with self._lock:
            # Perform a deep merge
            # This is simplified. A real update would be more careful,
            # especially with nested dicts like 'positions'.
            self._state["portfolio"].update(portfolio_update)
            logger.info("Portfolio state updated.")

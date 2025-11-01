import pandas as pd
from typing import Dict, Any, Optional

from monitor.logging import get_logger
from data_manager import DataManager
from core.pipeline_state import PipelineState
from core.schemas.data_schema import MarketEvent

# FIX: Renamed SimpleFeatureStore to FeatureStore
from features.store import FeatureStore
from cognitive.engine import CognitiveEngine

class BaseStrategy:
    """
    Abstract base class for all trading strategies.
    Defines the interface for event handling and signal generation.
    """
    def __init__(self, config: Dict[str, Any], data_manager: DataManager):
        self.config = config
        self.data_manager = data_manager
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info(f"Strategy '{self.__class__.__name__}' initialized.")

    async def on_event(self, event: MarketEvent, state: PipelineState) -> Optional[Dict[str, Any]]:
        """
        Asynchronously process an incoming market event.
        
        Args:
            event: The MarketEvent object (e.g., news, price update).
            state: The current PipelineState object.
            
        Returns:
            A dictionary representing a trading signal, or None if no action.
            Example signal: {"action": "BUY", "symbol": "AAPL", "weight": 0.1}
        """
        raise NotImplementedError("Strategy must implement on_event")

    async def on_decision_cycle(self, current_time: pd.Timestamp, state: PipelineState) -> Optional[Dict[str, Any]]:
        """
        Asynchronously triggered on a regular cycle (e.g., daily, hourly) 
        for portfolio rebalancing decisions.
        
        Args:
            current_time: The timestamp of the current decision cycle.
            state: The current PipelineState object.
            
        Returns:
            A dictionary representing a desired portfolio state or list of signals.
        """
        self.logger.debug(f"Decision cycle triggered at {current_time}")
        # Default implementation does nothing
        return None


class RomanLegionStrategy(BaseStrategy):
    """
    A sophisticated strategy handler that uses a "Cognitive Engine" to make
    decisions. It represents one "Legion" of the trading system.
    """
    
    # FIX: Removed redundant data arguments (asset_analysis_data, sentiment_data)
    # The CognitiveEngine will now get this data via the DataManager.
    def __init__(self, config: Dict[str, Any], data_manager: DataManager):
        """
        Initializes the strategy, its feature store, and the core cognitive engine.
        
        Args:
            config: The main system configuration dictionary.
            data_manager: The shared DataManager instance.
        """
        super().__init__(config, data_manager)
        
        # FIX: Renamed SimpleFeatureStore to FeatureStore
        self.feature_store = FeatureStore()
        self.logger.info("FeatureStore initialized.")

        # FIX: Passed config and data_manager to CognitiveEngine.
        # The CognitiveEngine is now self-sufficient and will load its own data
        # via the DataManager as needed.
        self.cognitive_engine = CognitiveEngine(
            config=self.config, 
            data_manager=self.data_manager
        )
        
        # Get references to the engine's components for convenience
        self.portfolio_constructor = self.cognitive_engine.portfolio_constructor
        self.risk_manager = self.cognitive_engine.risk_manager
        
        self.logger.info("RomanLegionStrategy initialized with CognitiveEngine.")

    async def on_event(self, event: MarketEvent, state: PipelineState) -> Optional[Dict[str, Any]]:
        """
        Processes high-priority, real-time events (e.g., breaking news).
        This path is for rapid, tactical decisions.
        """
        self.logger.debug(f"Processing event: {event.event_id} ({event.event_type})")
        
        # 1. Update features based on the event
        new_features = self.feature_store.update_features(event)
        
        # 2. (Optional) Quick check for immediate action
        # This bypasses the full cognitive loop for speed
        if event.event_type == 'URGENT_NEWS' and 'AAPL' in event.symbols:
             # Example: A simple heuristic rule
             if "positive guidance" in event.content.lower():
                 self.logger.info("Tactical BUY signal triggered by urgent news.")
                 return {"action": "TACTICAL_BUY", "symbol": "AAPL", "weight": 0.02}

        # 3. For most events, we just log and wait for the main decision cycle
        self.logger.debug(f"Event {event.event_id} logged. Awaiting decision cycle.")
        return None

    async def on_decision_cycle(self, current_time: pd.Timestamp, state: PipelineState) -> Optional[Dict[str, Any]]:
        """
        This is the main entry point for the full cognitive workflow.
        It runs the entire RAG, reasoning, and portfolio construction process.
        """
        self.logger.info(f"--- RomanLegionDecisionCycle START: {current_time} ---")
        
        try:
            # 1. Define the primary analysis task for this cycle
            # In a real system, this would be more dynamic (e.g., top 50 assets)
            task_description = "Analyze the market outlook for AAPL and GOOGL for the next 5 trading days."
            
            # 2. Run the full cognitive engine workflow
            # This is an async call that performs RAG, multi-agent reasoning,
            # synthesis, risk assessment, and portfolio construction.
            portfolio_decision = await self.cognitive_engine.run_cycle(
                task_description=task_description,
                current_time=current_time,
                current_state=state
            )
            
            if portfolio_decision is None:
                self.logger.warning("Cognitive engine returned no decision.")
                return None

            # 3. Log and return the final decision
            # The decision object (e.g., PortfolioDecision) contains the
            # target weights, the reasoning, and any generated orders.
            self.logger.info(f"Cognitive engine produced decision: {portfolio_decision.decision_id}")
            
            # The orchestrator will receive this and pass it to the OrderManager
            return portfolio_decision.to_dict() 
            
        except Exception as e:
            self.logger.error(f"Error during decision cycle: {e}", exc_info=True)
            # Propagate error to orchestrator's error handler
            raise
        finally:
            self.logger.info(f"--- RomanLegionDecisionCycle END: {current_time} ---")

import asyncio
import pandas as pd
from typing import Dict, Any, List, Optional

from ..monitor.logging import get_logger
from ..core.pipeline_state import PipelineState
from ..core.schemas.data_schema import MarketEvent, TickerData
from ..core.schemas.fusion_result import FusionResult
from ..context_bus import ContextBus
from ..strategy_handler import StrategyDataHandler
from ..cognitive.engine import CognitiveEngine
from ..execution.order_manager import OrderManager
from ..execution.trade_lifecycle_manager import TradeLifecycleManager
from ..audit_manager import AuditManager
from .error_handler import ErrorHandler

logger = get_logger(__name__)

class Orchestrator:
    """
    The central coordinator of the entire system.
    It owns the "master" copy of the PipelineState and is responsible for:
    1. Receiving events (from StreamProcessor or Backtester).
    2. Updating the state (via StrategyDataHandler).
    3. Triggering the CognitiveEngine.
    4. Passing the resulting Signal to the OrderManager.
    5. Updating the portfolio (via TradeLifecycleManager).
    6. Logging everything (via AuditManager).
    7. Updating the global ContextBus.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        context_bus: ContextBus,
        strategy_handler: StrategyDataHandler,
        cognitive_engine: CognitiveEngine,
        order_manager: OrderManager,
        trade_lifecycle_manager: TradeLifecycleManager,
        audit_manager: AuditManager,
        error_handler: ErrorHandler
    ):
        self.config = config
        self.context_bus = context_bus
        self.strategy_handler = strategy_handler
        self.cognitive_engine = cognitive_engine
        self.order_manager = order_manager
        self.trade_lifecycle_manager = trade_lifecycle_manager
        self.audit_manager = audit_manager
        self.error_handler = error_handler
        
        # The Orchestrator holds the "master" state
        self.master_state: PipelineState = context_bus.get_current_state()
        
        self.processing_lock = asyncio.Lock() # Ensures one event at a time
        logger.info("Orchestrator initialized.")

    async def preload_historical_data(self, end_date: pd.Timestamp):
        """(Backtesting) Preloads historical data into the StrategyDataHandler."""
        try:
            await self.strategy_handler.preload_historical_data(end_date)
            # Sync the master state
            self.master_state = self.strategy_handler.get_strategy_state(end_date)
            self.context_bus.update_state(self.master_state)
            logger.info(f"Historical data preloaded up to {end_date}.")
        except Exception as e:
            self.error_handler.handle_error(e, "preload_historical_data", True, self.audit_manager)

    async def update_state_from_batch(self, timestamp: pd.Timestamp, events_batch: List[Dict]):
        """(Backtesting) Updates state from a batch of timed events."""
        # This is a simplified method for backtesting
        # A real-time system would process events as they arrive
        try:
            for event_data in events_batch:
                if event_data.get('type') == 'MarketEvent':
                    # TODO: This requires DataAdapter to be accessible
                    # event = self.data_adapter.adapt_news_event(event_data)
                    # if event: self.strategy_handler.update_market_event(event)
                    pass
                elif event_data.get('type') == 'TickerData':
                    # ticker = self.data_adapter.adapt_market_data(event_data)
                    # if ticker: self.strategy_handler.update_market_data(ticker)
                    pass
            
            # After processing all events in the batch, get the new state
            self.master_state = self.strategy_handler.get_strategy_state(timestamp)
            self.context_bus.update_state(self.master_state)
            
        except Exception as e:
            self.error_handler.handle_error(e, "update_state_from_batch", False, self.audit_manager)

    async def process_event(self, event: MarketEvent):
        """
        (Live/API) Entry point for a single, new MarketEvent.
        This triggers the full cognitive and execution pipeline.
        """
        logger.info(f"Received event for processing: {event.event_id}")
        
        # Log the raw event to the audit trail
        self.audit_manager.log_event_in(event.model_dump())
        
        # Ensure only one event is processed at a time
        async with self.processing_lock:
            # Check circuit breaker *before* processing
            if self.error_handler.circuit_breaker.is_open():
                logger.error(f"Circuit breaker is OPEN. Skipping event {event.event_id}.")
                return
            
            try:
                # 1. Update State
                self.strategy_handler.update_market_event(event)
                # We also need to update price data
                # TODO: Add logic to fetch latest TickerData for event.symbols
                
                # Get the new "master state" snapshot
                self.master_state = self.strategy_handler.get_strategy_state(
                    timestamp=event.timestamp,
                    triggering_event=event
                )
                
                # 2. Run Cognitive Pipeline
                fusion_result = await self.cognitive_engine.run_cognitive_pipeline(self.master_state)
                
                if not fusion_result:
                    raise ValueError("Cognitive pipeline returned no result.")
                
                # Log all AI I/O and the final decision
                self.audit_manager.log_pipeline_run(fusion_result)

                # 3. Handle ERROR/INVALID states from AI
                if fusion_result.status != "SUCCESS":
                    raise ValueError(f"Cognitive pipeline failed: {fusion_result.error_message}")
                
                # 4. Generate Target Signal
                signal = self.cognitive_engine.generate_target_signal(self.master_state, fusion_result)
                self.audit_manager.log_signal(signal)
                
                # 5. Generate Orders
                orders = self.order_manager.generate_orders_from_signal(signal, self.master_state)
                
                # 6. Simulate/Execute Trades
                fills, costs = self.order_manager.simulate_execution(orders, self.master_state)
                
                # 7. Update Portfolio State (Master State)
                self.trade_lifecycle_manager.update_portfolio_with_fills(
                    fills, costs, self.master_state.timestamp
                )
                
                # 8. Sync the master state with the new portfolio
                self.master_state.portfolio = self.trade_lifecycle_manager.get_current_portfolio()
                
                # 9. Update global context
                self.context_bus.update_state(self.master_state)
                
                # 10. Record success for circuit breaker
                self.error_handler.circuit_breaker.record_success()
                
                logger.info(f"Successfully processed and executed event: {event.event_id}")
                
            except Exception as e:
                # Record failure for circuit breaker
                self.error_handler.handle_error(e, f"process_event:{event.event_id}", True, self.audit_manager)


    # --- Methods for Scheduled Tasks ---
    
    async def trigger_daily_training(self):
        """Triggers the daily walk-forward training process."""
        logger.info("Orchestrator: Received trigger for daily training.")
        # TODO: Implement call to ai.walk_forward_trainer
        pass

    async def trigger_data_validation(self):
        """Triggers the hourly data validation check."""
        logger.info("Orchestrator: Received trigger for data validation.")
        # TODO: Implement call to scripts.validate_data
        pass

    # --- Utility Methods ---
    
    def get_latest_prices(self) -> Dict[str, float]:
        """Gets the latest prices from the master state."""
        prices = {}
        for symbol, data_list in self.master_state.market_data.items():
            if data_list:
                prices[symbol] = data_list[-1].close # Get last close price
        return prices
        
    def get_status(self) -> Dict[str, Any]:
        """Gets the current status of the orchestrator for the API."""
        return {
            "status": "running",
            "mode": self.config.get('system_mode', 'simulation'),
            "last_event_time": self.master_state.timestamp.isoformat(),
            "circuit_breaker_state": self.error_handler.circuit_breaker.state,
            "circuit_breaker_failures": self.error_handler.circuit_breaker.failure_count
        }

"""
Orchestrator
The central "brain" of the Phoenix project.

Connects all components:
- Receives events from the EventDistributor (live) or Scheduler (batch).
- Manages the PipelineState (current time, portfolio, positions).
- Triggers the StrategyHandler (which contains the CognitiveEngine).
- Sends resulting signals to the OrderManager.
- Dispatches heavy tasks to the Celery Worker.
"""
import logging
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any, Union

# 修复：为所有本地模块添加 `..` 相对导入
from ..data_manager import DataManager
from ..core.pipeline_state import PipelineState
from ..core.schemas.data_schema import MarketEvent, EconomicEvent
from ..strategy_handler import RomanLegionStrategy # Assuming this is the main strategy
from ..execution.order_manager import OrderManager
from ..execution.signal_protocol import StrategySignal
from ..events.event_distributor import EventDistributor
from ..worker import app as celery_app
from ..config.loader import ConfigLoader

logger = logging.getLogger(__name__)

class Orchestrator:
    """
    Coordinates the flow of data and decision-making in the system.
    """
    def __init__(self, 
                 config_loader: ConfigLoader,
                 pipeline_state: PipelineState,
                 strategy_handler: RomanLegionStrategy,
                 data_manager: DataManager,
                 order_manager: OrderManager,
                 event_distributor: EventDistributor,
                 is_live: bool = False):
        
        self.config_loader = config_loader
        self.pipeline_state = pipeline_state
        self.strategy_handler = strategy_handler
        self.data_manager = data_manager
        self.order_manager = order_manager
        self.event_distributor = event_distributor # Used for *receiving* events
        self.is_live = is_live # Flag to differentiate live vs. backtest

        self.is_processing = asyncio.Event() # Lock to prevent concurrent processing
        logger.info("Orchestrator initialized.")

    async def on_event(self, event: Union[MarketEvent, EconomicEvent]):
        """
        Primary entry point for real-time events.
        Called by the EventDistributor's listener.
        """
        logger.debug(f"Received event: {event.event_id}")
        
        # Update pipeline state with the latest event time
        self.pipeline_state.update_time(event.timestamp)
        
        # 1. Update DataManager with the new event
        # This might involve saving it, updating features, etc.
        await self.data_manager.update_with_event(event)
        
        # 2. Trigger the cognitive workflow (asynchronously)
        # We don't want to block the event loop
        self.dispatch_cognitive_workflow(event)
        
    async def on_scheduled_task(self, task_name: str, trigger_time: datetime):
        """
        Entry point for scheduled tasks (e.g., daily rebalance).
        Called by the Scheduler.
        """
        logger.info(f"Received scheduled task: {task_name} at {trigger_time}")
        
        # Update pipeline state time
        self.pipeline_state.update_time(trigger_time)
        
        # 1. Update DataManager with latest market data (e.g., bar close)
        await self.data_manager.update_market_data(trigger_time)
        
        # 2. Trigger the cognitive workflow (asynchronously)
        self.dispatch_cognitive_workflow(event=None, task_name=task_name)
        
    def dispatch_cognitive_workflow(self, 
                                    event: Optional[Union[MarketEvent, EconomicEvent]] = None, 
                                    task_name: Optional[str] = None):
        """
        Checks the processing lock and dispatches the task to Celery.
        """
        if self.is_live:
            if self.is_processing.is_set():
                logger.warning(f"Already processing a workflow. Ignoring new trigger.")
                return
            
            logger.info("Setting processing lock and dispatching to worker...")
            self.is_processing.set() # Set the lock

            # Convert event to dict if it exists, as Celery needs serializable data
            event_dict = event.dict() if event else None
            
            # Send task to Celery worker
            # 修复：使用正确的 worker 引用
            celery_app.send_task(
                'worker.run_cognitive_workflow', # Name of the task in worker.py
                args=[event_dict, task_name],
                # Callback to release the lock *after* the task is done
                link=celery_app.signature('worker.release_processing_lock')
            )
        else:
            # In backtesting, we run this synchronously
            logger.debug("Running cognitive workflow synchronously (backtest mode).")
            # We need an async task to run the async workflow
            asyncio.create_task(self.run_cognitive_workflow_sync(event, task_name))
            
    async def run_cognitive_workflow_sync(self, event: Optional[Union[MarketEvent, EconomicEvent]], task_name: Optional[str]):
        """
        A synchronous wrapper for backtesting.
        """
        try:
            # 1. Run the strategy
            signal = await self.strategy_handler.on_event(
                event=event,
                current_time=self.pipeline_state.get_current_time()
            )
            
            # 2. Process the signal
            if signal:
                await self.process_signal(signal)
            
        except Exception as e:
            logger.error(f"Synchronous workflow failed: {e}", exc_info=True)
        finally:
            logger.debug("Synchronous workflow complete.")


    async def process_signal(self, signal: StrategySignal):
        """
        Called (in backtest) or as a callback (in live) to process the
        StrategySignal from the cognitive workflow.
        """
        try:
            logger.info(f"Processing signal for timestamp: {signal.timestamp}")
            
            # 1. Update portfolio state with the new target weights
            self.pipeline_state.update_target_weights(signal.target_weights)
            
            # 2. Generate orders based on the signal
            orders = await self.order_manager.generate_orders_from_signal(signal)
            
            if not orders:
                logger.info("Signal generated no new orders.")
                return
                
            # 3. (Live Mode) Dispatch orders to execution adapter
            # if self.is_live:
            #    await self.order_manager.dispatch_orders(orders)
            # 4. (Backtest Mode) Orders are handled by BacktestingEngine loop
            
            logger.info(f"Generated {len(orders)} orders from signal.")

        except Exception as e:
            logger.error(f"Failed to process signal: {e}", exc_info=True)
        finally:
            # This is where the lock would be released in a live system
            # if this was the final step.
            pass

    def release_lock(self):
        """
        Callback function for Celery to release the processing lock.
        """
        logger.info("Releasing processing lock.")
        self.is_processing.clear()

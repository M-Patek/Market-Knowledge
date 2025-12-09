"""
Phoenix_project/controller/loop_manager.py
[Phase 4 Task 1] Fix Backtest Loop Data Injection & Time Machine.
"""
import asyncio
import logging
import time
from typing import Optional, Dict, Any, List
from datetime import datetime
from omegaconf import DictConfig

from Phoenix_project.controller.orchestrator import Orchestrator
from Phoenix_project.data.data_iterator import DataIterator
from Phoenix_project.context_bus import ContextBus
from Phoenix_project.core.pipeline_state import PipelineState

logger = logging.getLogger(__name__)

class LoopManager:
    """
    Manages the main execution loops (Live or Backtest).
    Controls the pulse of the system.
    """

    def __init__(
        self,
        orchestrator: Orchestrator,
        context_bus: ContextBus,
        loop_config: DictConfig,
        data_iterator: Optional[DataIterator] = None,
        pipeline_state: Optional[PipelineState] = None
    ):
        self.orchestrator = orchestrator
        self.context_bus = context_bus
        self.config = loop_config
        self.data_iterator = data_iterator
        
        # State management: In backtest, we hold the state locally to persist between ticks
        self.pipeline_state = pipeline_state or PipelineState(run_id=f"run_{int(time.time())}")
        self._running = False
        self._stop_event = asyncio.Event()

    async def start_loop(self):
        """
        Main entry point. Decides which loop to run based on config.
        """
        self._running = True
        mode = self.config.get("mode", "live").lower()
        
        logger.info(f"LoopManager starting in {mode.upper()} mode.")
        
        try:
            if mode == "backtest":
                await self._backtest_loop()
            else:
                await self._live_loop()
        except Exception as e:
            logger.critical(f"Loop crashed: {e}", exc_info=True)
        finally:
            self._running = False
            logger.info("LoopManager stopped.")

    def stop_loop(self):
        """Signals the loop to stop."""
        self._running = False
        self._stop_event.set()
        logger.info("Stop signal received.")

    async def _live_loop(self):
        """
        Live execution loop. Fetches data from Redis via Orchestrator default behavior.
        """
        interval = self.config.get("interval_seconds", 60)
        
        while self._running and not self._stop_event.is_set():
            try:
                # Live mode: Orchestrator fetches data from Redis
                # State is loaded from ContextBus inside Orchestrator
                await self.orchestrator.run_main_cycle()
                
                # Sleep for interval
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=interval)
                except asyncio.TimeoutError:
                    pass 
                    
            except Exception as e:
                logger.error(f"Error in live loop: {e}")
                await asyncio.sleep(5) 

    async def _backtest_loop(self):
        """
        Backtest execution loop.
        [Fix] Injects data directly into Orchestrator to bypass Redis.
        [Fix] Time Machine: Syncs PipelineState time with historical data.
        """
        if not self.data_iterator:
            logger.critical("Backtest mode requires a DataIterator.")
            return

        logger.info("Starting Backtest Loop...")
        total_steps = 0
        start_perf = time.time()
        
        # Iterate through historical data batches
        async for batch in self.data_iterator:
            if not self._running or self._stop_event.is_set():
                break
                
            try:
                # [Time Machine] 1. Extract timestamp from batch
                current_step_time = datetime.now() # Fallback
                
                # Check typical list of dicts structure
                if batch and isinstance(batch, list) and len(batch) > 0:
                    last_item = batch[-1]
                    if isinstance(last_item, dict):
                        # Try standard keys
                        ts = last_item.get('timestamp') or last_item.get('date') or last_item.get('time')
                        if ts:
                            if isinstance(ts, (int, float)):
                                current_step_time = datetime.fromtimestamp(ts)
                            elif isinstance(ts, str):
                                try: current_step_time = datetime.fromisoformat(ts)
                                except: pass
                            elif isinstance(ts, datetime):
                                current_step_time = ts
                    elif hasattr(last_item, 'timestamp'):
                        current_step_time = last_item.timestamp

                # [Time Machine] 2. Force Sync PipelineState Time
                self.pipeline_state.current_time = current_step_time
                self.pipeline_state.step_index = total_steps

                # [Critical Fix] Inject batch AND state directly!
                await self.orchestrator.run_main_cycle(
                    state=self.pipeline_state,
                    injected_data=batch
                )
                
                total_steps += 1
                if total_steps % 100 == 0:
                    logger.info(f"Backtest progress: Step {total_steps} (Sim Time: {current_step_time})")
                
                # Optional: Flow control simulation
                # await asyncio.sleep(0.001)
                
            except Exception as e:
                logger.error(f"Error in backtest step {total_steps}: {e}", exc_info=True)
                # Configurable stop on error
                if self.config.get("stop_on_error", True):
                    raise
        
        duration = time.time() - start_perf
        logger.info(f"Backtest Completed. Steps: {total_steps}, Duration: {duration:.2f}s")

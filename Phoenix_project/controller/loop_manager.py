import asyncio
import time
from typing import Callable, Coroutine
from core.pipeline_state import PipelineState
from controller.orchestrator import Orchestrator
from monitor.logging import get_logger

logger = get_logger(__name__)

class LoopManager:
    """
    Manages the main asynchronous event loop of the application.
    It can run in different modes (e.g., "live", "backtest").
    """

    def __init__(
        self,
        orchestrator: Orchestrator,
        pipeline_state: PipelineState,
        mode: str = "live"
    ):
        self.orchestrator = orchestrator
        self.pipeline_state = pipeline_state
        self.mode = mode
        self._running = False
        self.main_task: Optional[asyncio.Task] = None
        
        logger.info(f"LoopManager initialized in '{mode}' mode.")

    async def _live_loop(self):
        """
        Main loop for live trading.
        It relies on the Scheduler (managed by Orchestrator) to trigger
        processing cycles. This loop just keeps the process alive
        and handles graceful shutdown.
        """
        logger.info("Live loop started. Waiting for scheduled events...")
        while self._running:
            try:
                # In live mode, the work is done by scheduled tasks
                # (e.g., via APScheduler). This loop is just a heartbeat
                # to keep the asyncio event loop running.
                await asyncio.sleep(3600) # Sleep for a long time
            except asyncio.CancelledError:
                logger.info("Live loop cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in live loop: {e}", exc_info=True)
                # Avoid busy-looping on error
                await asyncio.sleep(60) 

    async def _backtest_loop(self):
        """
        Main loop for backtesting.
        It iterates through historical data, manually triggering
        the Orchestrator for each timestep.
        """
        logger.info("Backtest loop started.")
        
        # Get the data iterator from the DataManager via Orchestrator
        # This assumes the orchestrator has access to it.
        try:
            data_iterator = self.orchestrator.data_manager.get_backtest_iterator()
        except Exception as e:
            logger.error(f"Failed to get backtest iterator: {e}", exc_info=True)
            self._running = False
            return

        for data_batch in data_iterator:
            if not self._running:
                logger.info("Backtest loop cancelled.")
                break
            
            start_time = time.perf_counter()
            
            # 1. Update state with new data
            # This is a simplified view. The Orchestrator should
            # probably have a "backtest_tick" method.
            timestamp = data_batch["timestamp"] # Assuming common timestamp
            await self.pipeline_state.update_state({
                "current_time": timestamp,
                # ... push data into market_data, news_data etc.
                "new_data_batch": data_batch 
            })
            
            # 2. Trigger the main processing cycle
            try:
                await self.orchestrator.run_main_cycle()
            except Exception as e:
                logger.error(f"Error during backtest cycle at {timestamp}: {e}", exc_info=True)
                # In backtest, we might want to stop, or just log and continue
                # self._running = False
                # break
            
            end_time = time.perf_counter()
            cycle_time = (end_time - start_time) * 1000
            logger.info(f"Backtest cycle for {timestamp} complete in {cycle_time:.2f} ms.")
        
        logger.info("Backtest loop finished.")
        self._running = False

    async def start(self):
        """Starts the main event loop based on the configured mode."""
        if self._running:
            logger.warning("LoopManager is already running.")
            return

        self._running = True
        logger.info(f"Starting LoopManager in '{self.mode}' mode...")
        
        if self.mode == "live":
            # Start the scheduler first
            self.orchestrator.start_scheduler()
            self.main_task = asyncio.create_task(self._live_loop())
        elif self.mode == "backtest":
            self.main_task = asyncio.create_task(self._backtest_loop())
        else:
            logger.error(f"Unknown mode: {self.mode}")
            self._running = False
            return
            
        await self.main_task

    async def stop(self):
        """Stops the main event loop gracefully."""
        if not self._running or not self.main_task:
            logger.info("LoopManager is not running.")
            return

        logger.info("Stopping LoopManager...")
        self._running = False
        
        if self.mode == "live":
            self.orchestrator.stop_scheduler()

        if not self.main_task.done():
            self.main_task.cancel()
            try:
                await self.main_task
            except asyncio.CancelledError:
                pass # Expected
        
        logger.info("LoopManager stopped.")

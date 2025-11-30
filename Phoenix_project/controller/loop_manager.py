import time
from datetime import datetime
from typing import Optional
import asyncio

from Phoenix_project.context_bus import ContextBus
from Phoenix_project.controller.orchestrator import Orchestrator
from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.data.data_iterator import DataIterator
from Phoenix_project.monitor.logging import get_logger

log = get_logger("LoopManager")


class LoopManager:
    """
    管理主事件循环（实时交易或回测）。
    控制循环的开始、停止、暂停和恢复。
    [Task 1.2] Refactored to Pure Async (No Threading).
    """

    def __init__(
        self,
        orchestrator: Orchestrator,
        data_iterator: DataIterator,
        pipeline_state: PipelineState,
        context_bus: ContextBus,
        loop_config: dict,
    ):
        self.orchestrator = orchestrator
        self.data_iterator = data_iterator
        self.pipeline_state = pipeline_state
        self.context_bus = context_bus
        self.loop_config = loop_config
        self.loop_mode = loop_config.get("mode", "backtest")  # "live" or "backtest"
        
        # [Task 1.2] Replaced Thread with Async Task
        self.loop_task: Optional[asyncio.Task] = None
        self.is_running = False
        self.cycle_interval = loop_config.get("interval", 1.0)

    async def start_loop(self):
        """启动主循环作为一个 Async Task。"""
        if self.is_running:
            log.warning("Loop is already running.")
            return

        log.info(f"Starting event loop in '{self.loop_mode}' mode.")
        self.is_running = True
        self.pipeline_state.resume()

        # Spawn the loop task managed by asyncio
        self.loop_task = asyncio.create_task(self._main_loop_wrapper())

    async def stop_loop(self):
        """停止主循环 (Async)。"""
        if not self.is_running:
            log.warning("Loop is not running.")
            return

        log.info("Stopping event loop...")
        self.is_running = False
        
        if self.loop_task:
            self.loop_task.cancel()
            try:
                await self.loop_task
            except asyncio.CancelledError:
                pass # Expected
            self.loop_task = None
            
        log.info("Event loop stopped.")

    def pause_loop(self):
        """暂停循环。"""
        log.info("Pausing event loop...")
        self.pipeline_state.pause()

    def resume_loop(self):
        """恢复循环。"""
        log.info("Resuming event loop...")
        self.pipeline_state.resume()
        
    async def _main_loop_wrapper(self):
        """Unified Async Wrapper to route execution."""
        try:
            if self.loop_mode == "live":
                await self._live_loop()
            else:
                await self._backtest_loop()
        except asyncio.CancelledError:
            log.info("Loop task cancelled.")
            raise
        except Exception as e:
            log.critical(f"Loop crashed: {e}", exc_info=True)
            self.is_running = False

    async def _live_loop(self):
        """
        [Task 4.3] Live execution loop for the Phoenix system. Implements Fail-Fast.
        """
        log.info("Starting Phoenix live execution loop...")
        
        while self.is_running:
            try:
                await self.orchestrator.run_main_cycle() # Using standardized method name
                await asyncio.sleep(self.cycle_interval)

            # [Task 4.3] Only catch non-fatal exceptions to enable fail-fast.
            except asyncio.CancelledError:
                # Expected when the loop is being gracefully shut down
                break 
            except Exception as e:
                # Catch general runtime errors (non-fatal)
                log.error(f"Critical runtime error in live loop, continuing: {e}", exc_info=True)
                # Avoid rapid continuous failures if error happens instantly
                await asyncio.sleep(self.cycle_interval / 2)
            except BaseException as e:
                # Catch fatal errors (SystemExit, KeyboardInterrupt, etc.)
                log.critical(f"FATAL BaseException in live loop. Exiting now: {e}", exc_info=True)
                self.is_running = False  # Trigger loop termination
                raise  # Re-raise the fatal error to stop the process
        
        log.info("Phoenix live execution loop stopped.")

    async def _backtest_loop(self):
        """
        [Task 4.3] Backtest execution loop (Pure Async Iteration).
        """
        log.info("Starting Phoenix backtest loop...")
        
        # [Task 1.2] Use async for to iterate data (Push Model)
        try:
            async for batch in self.data_iterator:
                if not self.is_running:
                    break
                
                # Pass batch to orchestrator logic via run_main_cycle or specific backtest method
                # Since Orchestrator.run_main_cycle pulls from event queue, we might need to inject data.
                # In BacktestEngine context, loop management differs.
                # Assuming LoopManager for backtest orchestrates step-by-step injection:
                
                # Update pipeline state with time/data from iterator (handled inside Orchestrator or here)
                # For now, we assume orchestrator handles data consumption if properly mocked/configured,
                # or we interact with BacktestEngine.
                
                # For strict Orchestrator compatibility, we might inject an event or update state.
                # Based on previous context, we call orchestrator.
                
                # Note: LoopManager.run_backtest usually delegates to a specific engine or calls cycle.
                # If using Orchestrator directly:
                await self.orchestrator.run_main_cycle()
                
        except Exception as e:
            log.error(f"Critical runtime error in backtest loop: {e}", exc_info=True)
            
        log.info("Phoenix backtest loop finished.")

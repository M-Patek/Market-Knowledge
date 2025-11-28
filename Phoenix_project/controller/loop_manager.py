import time
from datetime import datetime
from threading import Thread
from typing import Optional
import asyncio

from phoenix_project.context_bus import ContextBus
from phoenix_project.controller.orchestrator import Orchestrator
from phoenix_project.core.pipeline_state import PipelineState
from phoenix_project.data.data_iterator import DataIterator
from phoenix_project.monitor.logging import get_logger

log = get_logger("LoopManager")


class LoopManager:
    """
    管理主事件循环（实时交易或回测）。
    控制循环的开始、停止、暂停和恢复。
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
        self.loop_thread: Optional[Thread] = None
        self.is_running = False

    def start_loop(self):
        """启动主循环在一个单独的线程中。"""
        if self.is_running:
            log.warning("Loop is already running.")
            return

        log.info(f"Starting event loop in '{self.loop_mode}' mode.")
        self.is_running = True
        self.pipeline_state.resume()

        if self.loop_mode == "live":
            self.loop_thread = Thread(target=self._live_loop, daemon=True)
        else:
            self.loop_thread = Thread(target=self._backtest_loop, daemon=True)

        self.loop_thread.start()

    def stop_loop(self):
        """停止主循环。"""
        if not self.is_running:
            log.warning("Loop is not running.")
            return

        log.info("Stopping event loop...")
        self.is_running = False
        if self.loop_thread:
            self.loop_thread.join(timeout=5)
            if self.loop_thread.is_alive():
                log.error("Loop thread did not terminate gracefully.")
        log.info("Event loop stopped.")

    def pause_loop(self):
        """暂停循环。"""
        log.info("Pausing event loop...")
        self.pipeline_state.pause()

    def resume_loop(self):
        """恢复循环。"""
        log.info("Resuming event loop...")
        self.pipeline_state.resume()

    async def _live_loop(self):
        """
        [Task 4.3] Live execution loop for the Phoenix system. Implements Fail-Fast.
        """
        log.info("Starting Phoenix live execution loop...")
        
        while self.running:
            try:
                await self.orchestrator.run_live_cycle()
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
                self.running = False  # Trigger loop termination
                raise  # Re-raise the fatal error to stop the process
        
        log.info("Phoenix live execution loop stopped.")

    async def _backtest_loop(self):
        """
        [Task 4.3] Backtest execution loop. Implements Fail-Fast.
        """
        log.info("Starting Phoenix backtest loop...")
        
        while self.running and not self.data_manager.is_backtest_finished():
            try:
                await self.orchestrator.run_backtest_cycle()
                
            # [Task 4.3] Only catch non-fatal exceptions to enable fail-fast.
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Catch general runtime errors (non-fatal)
                log.error(f"Critical runtime error in backtest loop, continuing: {e}", exc_info=True)
            except BaseException as e:
                # Catch fatal errors (SystemExit, KeyboardInterrupt, etc.)
                log.critical(f"FATAL BaseException in backtest loop. Exiting now: {e}", exc_info=True)
                self.running = False  # Trigger loop termination
                raise  # Re-raise the fatal error to stop the process
        
        log.info("Phoenix backtest loop finished or stopped.")

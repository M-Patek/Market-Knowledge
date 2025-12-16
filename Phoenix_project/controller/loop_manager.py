import asyncio
import logging
import time
from datetime import datetime
from typing import Optional, Callable, Any

logger = logging.getLogger(__name__)

class LoopManager:
    """
    Control loop manager that ensures periodic execution with drift correction.
    """
    def __init__(self, interval: float, bus: Any = None):
        self.interval = interval
        self.bus = bus
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self, work_function: Callable):
        """Starts the main execution loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run_loop(work_function))
        logger.info(f"LoopManager started with interval {self.interval}s.")

    async def stop(self):
        """Stops the loop gracefully."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("LoopManager stopped.")

    # Legacy alias support if needed
    async def start_loop(self, work_function: Callable = None):
        if work_function:
            await self.start(work_function)
    
    def stop_loop(self):
        # Fire and forget cancel for sync context calling
        if self._task:
            self._task.cancel()
        self._running = False

    async def _run_loop(self, work_function: Callable):
        """
        [Task P1-DATA-02] Drift-Corrected Loop Implementation.
        Uses a target timestamp to calculate sleep duration, preventing time drift accumulation.
        """
        loop = asyncio.get_running_loop()
        # Initialize the target time to 'now'
        next_tick = loop.time()

        while self._running:
            try:
                # 1. Advance the target clock by one interval
                next_tick += self.interval
                
                # 2. Execute the main work payload
                start_time = datetime.now()
                if asyncio.iscoroutinefunction(work_function):
                    await work_function()
                else:
                    work_function()
                    
                work_duration = (datetime.now() - start_time).total_seconds()

                # [Task P1-DATA-02] Emit Heartbeat
                if self.bus:
                    payload = {
                        "type": "heartbeat",
                        "component": "LoopManager",
                        "timestamp": datetime.now().isoformat(),
                        "status": "running",
                        "interval": self.interval,
                        "last_cycle_duration": work_duration,
                        "drift_lag": max(0, work_duration - self.interval) # Approximated lag indication
                    }
                    # Fire and forget heartbeat to avoid blocking
                    asyncio.create_task(self.bus.publish("SYSTEM_HEARTBEAT", payload))

                # 3. Calculate sleep time to align with next_tick
                now = loop.time()
                sleep_duration = next_tick - now

                if sleep_duration > 0:
                    await asyncio.sleep(sleep_duration)
                else:
                    # System is overloaded or work took too long
                    lag = abs(sleep_duration)
                    logger.warning(f"LoopManager: Cycle overrun by {lag:.4f}s. Skipping sleep to catch up.")
                    
                    # Safety: If lag is massive (> 3 intervals), reset the clock to avoid 'fast-forwarding' bursts
                    if lag > self.interval * 3:
                        logger.error("LoopManager: Severe lag detected. Resetting loop clock.")
                        next_tick = loop.time()

            except asyncio.CancelledError:
                logger.info("LoopManager: Task cancelled.")
                break
            except Exception as e:
                logger.error(f"LoopManager: Unexpected error in loop: {e}", exc_info=True)
                # Sleep briefly on error to prevent CPU spinning
                await asyncio.sleep(1.0)

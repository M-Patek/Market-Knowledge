import asyncio
import signal
from typing import Dict, Any

from ..monitor.logging import get_logger
from ..events.stream_processor import StreamProcessor
from .scheduler import Scheduler
from ..interfaces.api_server import APIServer # Using api_server, not gateway
from .error_handler import ErrorHandler

logger = get_logger(__name__)

class LoopManager:
    """
    Manages the main asynchronous event loops for the application.
    It is responsible for starting, supervising, and gracefully
    shutting down all core components (e.g., StreamProcessor, Scheduler, API).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        stream_processor: StreamProcessor,
        scheduler: Scheduler,
        api_server: APIServer,
        error_handler: ErrorHandler
    ):
        """
        Initializes the LoopManager.
        
        Args:
            config: The main system configuration.
            stream_processor: The real-time event ingestion component.
            scheduler: The cron-like job scheduler component.
            api_server: The external API component.
            error_handler: The centralized error handler.
        """
        self.config = config
        self.stream_processor = stream_processor
        self.scheduler = scheduler
        self.api_server = api_server
        self.error_handler = error_handler
        
        self.tasks = []
        self._shutdown = asyncio.Event()

    def run(self):
        """
        Starts the main asyncio event loop and all managed components.
        This is the primary entry point for the application.
        """
        logger.info("Starting LoopManager and all application components...")
        try:
            asyncio.run(self.main())
        except (KeyboardInterrupt, SystemExit):
            logger.info("Shutdown signal received. Cleaning up...")
        except Exception as e:
            logger.critical(f"Unhandled exception in main loop: {e}", exc_info=True)
        finally:
            logger.info("LoopManager has shut down.")

    async def main(self):
        """
        The core async main function.
        """
        # 1. Setup signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._signal_handler, sig)
            
        # 2. Start all managed components as async tasks
        
        # Start the real-time StreamProcessor
        if self.config.get('enable_streaming', True):
            self.tasks.append(asyncio.create_task(
                self._run_component(self.stream_processor.run, "StreamProcessor")
            ))

        # Start the Scheduler in a separate thread (as it's blocking)
        if self.config.get('enable_scheduler', True):
            self.tasks.append(loop.run_in_executor(
                None, # Use default ThreadPoolExecutor
                self._run_blocking_component, 
                self.scheduler.run, 
                "Scheduler"
            ))

        # Start the API Server
        if self.config.get('enable_api', True):
            self.tasks.append(asyncio.create_task(
                self._run_component(self.api_server.run, "APIServer")
            ))
            
        logger.info(f"Started {len(self.tasks)} main components. Waiting for shutdown signal...")
        
        # 3. Wait for shutdown signal
        await self._shutdown.wait()
        
        # 4. Graceful shutdown
        logger.info("Initiating graceful shutdown of components...")
        
        # Signal blocking components to stop
        if self.scheduler:
            self.scheduler.stop()
        if self.stream_processor:
            self.stream_processor.stop()
        if self.api_server:
            self.api_server.stop() # Assuming API server has a stop() method
            
        # Wait for all tasks to complete
        await asyncio.gather(*self.tasks, return_exceptions=True)
        logger.info("All components shut down gracefully.")

    async def _run_component(self, component_run_func, name: str):
        """Wrapper to run an async component and handle errors."""
        try:
            logger.info(f"{name} starting...")
            await component_run_func()
        except asyncio.CancelledError:
            logger.info(f"{name} was cancelled (shutdown).")
        except Exception as e:
            logger.error(f"{name} failed: {e}", exc_info=True)
            self.error_handler.handle_error(e, name, is_critical=True)
            self._signal_handler(signal.SIGTERM) # Trigger system shutdown
        finally:
            logger.info(f"{name} stopped.")

    def _run_blocking_component(self, component_run_func, name: str):
        """Wrapper to run a blocking component (like Scheduler) in a thread."""
        try:
            logger.info(f"{name} starting in background thread...")
            component_run_func()
        except Exception as e:
            logger.error(f"{name} (blocking) failed: {e}", exc_info=True)
            # Can't trigger async signal handler from here easily,
            # so we just log the critical error.
            self.error_handler.handle_error(e, name, is_critical=True)
        finally:
            logger.info(f"{name} (blocking) stopped.")
            
    def _signal_handler(self, sig):
        """Sets the shutdown event when a signal is received."""
        logger.info(f"Received signal {sig.name}. Setting shutdown event.")
        self._shutdown.set()

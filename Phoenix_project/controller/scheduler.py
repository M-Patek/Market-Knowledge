import schedule
import time
import threading
from typing import Dict, Any

from .orchestrator import Orchestrator
from ..monitor.logging import get_logger

logger = get_logger(__name__)

class Scheduler:
    """
    Manages all scheduled (cron-like) tasks within the Phoenix system.
    This is distinct from the real-time event loop.
    
    Example tasks:
    - Daily model retraining
    - Weekly data integrity checks
    - Nightly cache clearing
    """

    def __init__(self, orchestrator: Orchestrator, config: Dict[str, Any]):
        """
        Initializes the Scheduler.
        
        Args:
            orchestrator (Orchestrator): The main orchestrator to trigger tasks on.
            config (Dict[str, Any]): The main system configuration.
        """
        self.orchestrator = orchestrator
        self.config = config.get('scheduler', {})
        self.jobs = []
        self._stop_run_continuously = threading.Event()

    def setup_schedules(self):
        """
        Reads the 'scheduler' config and sets up 'schedule' library jobs.
        """
        logger.info("Setting up scheduled jobs...")
        
        if not self.config:
            logger.warning("No 'scheduler' config found. No jobs will be scheduled.")
            return

        # Example: Schedule a daily "walk-forward training" task
        if self.config.get('enable_daily_training', False):
            train_time = self.config.get('daily_training_time', '01:00')
            logger.info(f"Scheduling daily training job at {train_time}")
            job = schedule.every().day.at(train_time).do(
                self.run_threaded, 
                self.orchestrator.trigger_daily_training
            )
            self.jobs.append(job)

        # Example: Schedule a data validation check
        if self.config.get('enable_hourly_validation', False):
            logger.info("Scheduling hourly data validation job")
            job = schedule.every().hour.do(
                self.run_threaded,
                self.orchestrator.trigger_data_validation
            )
            self.jobs.append(job)
            
        logger.info(f"Scheduled {len(self.jobs)} jobs.")

    def run_threaded(self, job_func, *args, **kwargs):
        """
        Wrapper to run a scheduled job in its own daemon thread.
        This prevents a long-running job (like training) from blocking
        the scheduler loop itself.
        """
        logger.info(f"Starting scheduled job: {job_func.__name__}")
        job_thread = threading.Thread(target=job_func, args=args, kwargs=kwargs, daemon=True)
        job_thread.start()

    def run(self):
        """
        Starts the main scheduler loop.
        This is a blocking call and should be run in its own thread or process.
        """
        logger.info("Scheduler started. Waiting for jobs...")
        self.setup_schedules()
        
        while not self._stop_run_continuously.is_set():
            try:
                schedule.run_pending()
                time.sleep(1) # Check for pending jobs every second
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}", exc_info=True)
                time.sleep(60) # Back off for a minute if loop fails

        logger.info("Scheduler shutting down.")

    def stop(self):
        """Signals the scheduler loop to stop."""
        logger.info("Received stop signal for scheduler.")
        self._stop_run_continuously.set()

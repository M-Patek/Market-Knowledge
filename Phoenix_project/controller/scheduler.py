from typing import Dict, Any, TYPE_CHECKING
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.asyncio import AsyncIOExecutor
from pytz import utc
from Phoenix_project.monitor.logging import get_logger

if TYPE_CHECKING:
    from Phoenix_project.controller.orchestrator import Orchestrator

logger = get_logger(__name__)

class Scheduler:
    """
    Manages all scheduled tasks for the application using APScheduler.
    This is primarily used in "live" mode to trigger the main
    processing cycle, data ingestion, etc.
    """

    def __init__(self, orchestrator: "Orchestrator", config: Dict[str, Any]):
        self.orchestrator = orchestrator
        self.config = config
        
        jobstores = {'default': MemoryJobStore()}
        executors = {'default': AsyncIOExecutor()}
        job_defaults = {'coalesce': True, 'max_instances': 1}
        
        self.scheduler = AsyncIOScheduler(
            jobstores=jobstores,
            executors=executors,
            job_defaults=job_defaults,
            timezone=utc
        )
        logger.info("Scheduler initialized.")

    def setup_jobs(self):
        """
        Loads job definitions from the config and adds them to the scheduler.
        """
        jobs = self.config.get("jobs", [])
        if not jobs:
            logger.warning("No jobs found in scheduler config.")
            return

        for job_config in jobs:
            try:
                job_id = job_config["id"]
                trigger_type = job_config["trigger"]["type"]
                trigger_args = job_config["trigger"]["args"]
                
                # Resolve the function to call
                # This is a simple mapper. A real system might use
                # dynamic imports or a registration pattern.
                if job_config["func"] == "orchestrator.run_main_cycle":
                    job_func = self.orchestrator.run_main_cycle
                elif job_config["func"] == "orchestrator.schedule_data_ingestion":
                    # This method needs to exist on the orchestrator
                    # job_func = self.orchestrator.schedule_data_ingestion
                    logger.warning("Job 'orchestrator.schedule_data_ingestion' not implemented.")
                    continue
                else:
                    logger.error(f"Unknown job function: {job_config['func']}")
                    continue
                    
                self.scheduler.add_job(
                    job_func,
                    trigger=trigger_type,
                    id=job_id,
                    **trigger_args
                )
                logger.info(f"Scheduled job '{job_id}' ({job_config['func']}) with trigger: {trigger_type} {trigger_args}")
                
            except Exception as e:
                logger.error(f"Failed to schedule job {job_config.get('id')}: {e}", exc_info=True)

    def start(self):
        """Starts the scheduler."""
        try:
            self.setup_jobs()
            self.scheduler.start()
            logger.info("APScheduler started.")
        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}", exc_info=True)

    def stop(self):
        """Stops the scheduler gracefully."""
        try:
            if self.scheduler.running:
                self.scheduler.shutdown()
                logger.info("APScheduler shut down.")
        except Exception as e:
            logger.error(f"Error shutting down scheduler: {e}", exc_info=True)

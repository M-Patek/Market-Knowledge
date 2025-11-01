import time
import logging
from typing import Dict, Any, Callable

# 修正：导入 'schedule' 库，这个库之前缺失了
import schedule

from .orchestrator import Orchestrator
from .error_handler import ErrorHandler

logger = logging.getLogger(__name__)

class Scheduler:
    """
    一个基于 'schedule' 库的 cron 式作业调度器。
    它用于触发周期性任务，例如每日数据摄取或
    定期的认知循环。
    """

    def __init__(
        self,
        config: Dict[str, Any],
        orchestrator: Orchestrator,
        error_handler: ErrorHandler
    ):
        """
        初始化调度器。
        
        Args:
            config: 'scheduler' 部分的配置。
            orchestrator: 用于调度作业的协调器实例。
            error_handler: 用于报告作业执行错误的处理器。
        """
        self.config = config.get('scheduler', {})
        self.orchestrator = orchestrator
        self.error_handler = error_handler
        self._stop_run_continuously = False
        logger.info("Scheduler initialized.")

    def setup_jobs(self):
        """
        从配置中读取并设置所有调度的作业。
        """
        jobs = self.config.get('jobs', [])
        logger.info(f"Setting up {len(jobs)} scheduled jobs...")
        
        for job in jobs:
            try:
                job_name = job.get('name')
                schedule_time = job.get('schedule_time') # e.g., "10:30"
                job_type = job.get('type') # e.g., "run_cognitive_workflow"
                job_params = job.get('params', {})
                
                if not all([job_name, schedule_time, job_type]):
                    logger.warning(f"Skipping incomplete job definition: {job}")
                    continue

                # 将作业类型映射到 Orchestrator 上的一个方法
                job_func: Callable[..., Any]
                if job_type == "run_cognitive_workflow":
                    # 创建一个 partial function 或 lambda 来包装带参数的调用
                    job_func = lambda params=job_params: self.orchestrator.schedule_cognitive_workflow(
                        task_description=params.get('task_description', 'Scheduled analysis'),
                        context=params
                    )
                elif job_type == "run_data_ingestion":
                    job_func = lambda params=job_params: self.orchestrator.schedule_data_ingestion(
                        sources=params.get('sources', ['all'])
                    )
                else:
                    logger.warning(f"Unknown job type '{job_type}' for job '{job_name}'. Skipping.")
                    continue
                
                # 使用 'schedule' 库设置作业
                # 示例：我们假设所有时间都是每日的
                logger.info(f"Scheduling job '{job_name}' ({job_type}) every day at {schedule_time}.")
                schedule.every().day.at(schedule_time).do(
                    self._job_wrapper, job_func, job_name
                )
                
            except Exception as e:
                logger.error(f"Failed to schedule job '{job.get('name')}': {e}", exc_info=True)
                self.error_handler.handle_error(e, f"schedule_job_{job.get('name')}")

    def _job_wrapper(self, job_func: Callable, job_name: str):
        """
        一个包装器，用于安全地执行调度的作业并处理错误。
        """
        logger.info(f"--- Running scheduled job: {job_name} ---")
        try:
            # 执行作业 (例如: orchestrator.schedule_cognitive_workflow(...))
            job_func()
            logger.info(f"--- Completed scheduled job: {job_name} ---")
        except Exception as e:
            logger.error(f"Scheduled job '{job_name}' failed: {e}", exc_info=True)
            self.error_handler.handle_error(e, f"run_job_{job_name}", is_critical=False)

    def run(self):
        """
        启动调度器的主循环 (这是一个阻塞调用)。
        """
        if not self.config.get('enabled', True):
            logger.info("Scheduler is disabled by config. Not starting.")
            return

        self.setup_jobs()
        logger.info("Scheduler loop started. Waiting for jobs...")
        
        self._stop_run_continuously = False
        while not self._stop_run_continuously:
            try:
                schedule.run_pending()
                time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Scheduler interrupted by user.")
                break
            except Exception as e:
                logger.error(f"Error in scheduler main loop: {e}", exc_info=True)
                self.error_handler.handle_error(e, "scheduler_loop", is_critical=True)
                time.sleep(60) # 发生故障时冷却 60 秒

        logger.info("Scheduler loop stopped.")

    def stop(self):
        """
        停止调度器的主循环。
        """
        logger.info("Stopping scheduler loop...")
        self._stop_run_continuously = True

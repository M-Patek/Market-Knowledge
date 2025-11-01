# Phoenix_project/controller/scheduler.py
import schedule
import asyncio
import time
import logging
from controller.orchestrator import Orchestrator

logging.basicConfig(level=logging.INFO)


def _run_job(task: dict):
    logging.info(f"Scheduler running task: {task.get('ticker', 'N/A')}")
    # 我们必须从这个同步上下文中运行异步 orchestrator
    asyncio.run(Orchestrator().run_pipeline(task))
    logging.info(f"Scheduler finished task: {task.get('ticker', 'N/A')}")


def schedule_jobs(cron_table: str) -> None:
    """Calls orchestrator.run_pipeline according to the configuration schedule."""
    # TODO: 实现从 cron_table (例如 daily_tasks.yaml) 动态加载作业
    # 示例占位符:
    schedule.every().day.at("08:00").do(_run_job, task={"task": "analyze NVDA stock", "ticker": "NVDA"})

    logging.info("Scheduler started. Waiting for jobs...")
    # while True:
    #     schedule.run_pending()
    #     time.sleep(1)
    pass # 我们不会在这里阻塞主线程；该模块将作为服务运行。

"""
Command-Line Interface (CLI) for interacting with the Phoenix system.

This script allows for:
- Manually triggering cognitive workflows.
- Injecting data.
- Checking system status.

Run from the root 'Phoenix_project' directory:
$ python scripts/run_cli.py --help
"""
import sys
import os
import asyncio
import logging
import argparse
from typing import Optional

# 修复：将项目根目录 (Phoenix_project) 添加到 sys.path
# 这允许脚本以 'python scripts/run_cli.py' 的方式运行
# 并正确解析 'from controller...' 或 'from core...'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 修复：现在使用绝对导入
from Phoenix_project.controller.orchestrator import Orchestrator
from Phoenix_project.worker import build_orchestrator
from Phoenix_project.core.schemas.data_schema import MarketEvent

# --- 日志设置 ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - [CLI] - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_orchestrator() -> Orchestrator:
    """
    Builds and returns an orchestrator instance for the CLI.
    This uses the same builder as the Celery worker.
    """
    logger.info("Building orchestrator instance for CLI...")
    # 注意：这会初始化所有依赖项（数据库、API 客户端等）
    try:
        orchestrator = build_orchestrator()
        return orchestrator
    except Exception as e:
        logger.error(f"Failed to build orchestrator: {e}", exc_info=True)
        logger.error("Please ensure all environment variables (DB URLs, API keys) are set.")
        sys.exit(1)

async def trigger_scheduled_task(task_name: str):
    """
    Simulates the triggering of a scheduled task.
    """
    logger.info(f"Manually triggering scheduled task: '{task_name}'")
    orchestrator = get_orchestrator()
    
    # 我们调用同步（回测）版本的工作流
    try:
        await orchestrator.run_cognitive_workflow_sync(event=None, task_name=task_name)
        logger.info(f"Task '{task_name}' executed successfully.")
    except Exception as e:
        logger.error(f"Error running task '{task_name}': {e}", exc_info=True)

async def inject_market_event(content: str, source: str):
    """
    Injects a new MarketEvent into the system.
    """
    logger.info(f"Injecting manual market event from '{source}'")
    orchestrator = get_orchestrator()

    # 1. 创建事件对象
    try:
        event_data = {
            "source": source,
            "timestamp": "now", # DataAdapter 会处理这个
            "content": content,
            "metadata": {"injected_by": "cli_tool"}
        }
        # 使用 DataAdapter 来标准化事件
        # (假设 orchestrator.data_manager 有一个 adapter)
        # 为了简单起见，我们直接创建它
        from datetime import datetime
        event = MarketEvent(
            id=f"cli-{datetime.utcnow().timestamp()}",
            source=source,
            timestamp=datetime.utcnow(),
            content=content,
            metadata={"injected_by": "cli_tool"}
        )
        logger.info(f"Created event: {event.id}")
    except Exception as e:
        logger.error(f"Failed to create MarketEvent object: {e}", exc_info=True)
        return

    # 2. 触发工作流
    try:
        await orchestrator.run_cognitive_workflow_sync(event=event, task_name=None)
        logger.info(f"Event '{event.id}' processed successfully.")
    except Exception as e:
        logger.error(f"Error processing injected event: {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(description="Phoenix Project CLI")
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # 'trigger' 命令
    trigger_parser = subparsers.add_parser("trigger", help="Trigger a scheduled task")
    trigger_parser.add_argument("task_name", 
                                type=str, 
                                help="The name of the task to trigger (e.g., 'daily_market_analysis')")
    
    # 'inject' 命令
    inject_parser = subparsers.add_parser("inject", help="Inject a manual market event")
    inject_parser.add_argument("-s", "--source", 
                               type=str, 
                               default="cli_manual", 
                               help="The source of the event")
    inject_parser.add_argument("content", 
                               type=str, 
                               help="The text content of the news or event")

    # 'status' 命令 (暂未实现)
    status_parser = subparsers.add_parser("status", help="Check system status (Not Implemented)")

    args = parser.parse_args()
    
    loop = asyncio.get_event_loop()

    if args.command == "trigger":
        loop.run_until_complete(trigger_scheduled_task(args.task_name))
    elif args.command == "inject":
        loop.run_until_complete(inject_market_event(args.content, args.source))
    elif args.command == "status":
        logger.info("System status check is not yet implemented.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

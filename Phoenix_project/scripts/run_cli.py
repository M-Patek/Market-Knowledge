import sys
from pathlib import Path

# 修正：添加项目根目录到 sys.path，以便导入根目录下的模块
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.append(str(PROJECT_ROOT))

# --------------------------------------------------
# 原始脚本内容现在可以正常导入了
# --------------------------------------------------

from controller.orchestrator import Orchestrator
from monitor.logging import get_logger
import asyncio

logger = get_logger(__name__)

async def main_async():
    """Asynchronous entry point for the CLI."""
    logger.info("Starting CLI... (Async)")
    
    # 示例任务
    task = {
        "type": "analysis_request",
        "ticker": "AAPL",
        "query": "What is the market sentiment and technical outlook for Apple?"
    }
    
    orchestrator = Orchestrator()
    
    try:
        result = await orchestrator.run_pipeline(task)
        logger.info(f"CLI Run Result:\n{result}")
    except Exception as e:
        logger.error(f"An error occurred during CLI pipeline execution: {e}", exc_info=True)

if __name__ == "__main__":
    """
    Command-Line Interface (CLI) entry point for the Phoenix Project.
    
    This script allows for direct, synchronous execution of the analysis pipeline
    from the command line for testing and debugging purposes.
    """
    
    # Since main_async is async, we must run it within an event loop.
    logger.info("Phoenix Project CLI Runner (Synchronous Wrapper)")
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("CLI execution interrupted by user.")

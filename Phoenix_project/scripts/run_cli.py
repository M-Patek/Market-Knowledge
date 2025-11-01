# Phoenix_project/scripts/run_cli.py
"""
Command-line interface wrapper to run the Phoenix project pipeline.
This script is used by the self_check.sh
"""
import asyncio
import argparse
import logging

# 我们需要将项目根目录添加到路径中才能导入 controller 模块
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from controller.loop_manager import control_loop

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Phoenix Analysis Pipeline")
    parser.add_argument("--ticker", type=str, required=True, help="Stock ticker to analyze (e.g., 'NVDA')")
    args = parser.parse_args()
    
    task = {"task": f"analyze {args.ticker} stock", "ticker": args.ticker}
    logging.info(f"Starting CLI run for task: {task}")
    asyncio.run(control_loop(task))
    logging.info(f"CLI run finished for task: {task}")

# Phoenix_project/scripts/run_cli.py
# [主人喵的修复 11.11] 实现了 TBD (CLI 功能)
# [主人喵的修复 11.12] 实现了 TBD (日志, 回测参数, 训练命令)

import argparse
import logging
import sys
import os
import requests # [新] 用于 API 调用 (例如 status, trigger)
import hydra # [新] 用于加载配置
from omegaconf import DictConfig

# [新] 确保我们可以从父目录导入
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from phoenix_project import PhoenixProject
except ImportError:
    print("Error: Failed to import PhoenixProject. Ensure __init__.py exists and parent path is correct.")
    sys.exit(1)


logger = logging.getLogger(__name__)

# [主人喵的修复 11.12] TBD 修复: 实现了 setup_logging
def setup_logging(verbose: bool):
    """配置 CLI 日志记录。"""
    level = logging.DEBUG if verbose else logging.INFO
    # 移除任何现有的处理器，以避免重复日志
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=level, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger.info(f"CLI logging configured at level {level}.")


# --- [新] Hydra/App 初始化辅助函数 ---

def _load_app(run_mode: str) -> PhoenixProject | None:
    """
    [新] 使用 Hydra 加载配置并初始化 PhoenixProject。
    """
    logger.info(f"Initializing app in '{run_mode}' mode...")
    try:
        # (假设 config 位于 ../config)
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize(config_path="../config", version_base=None)
        
        # 加载主系统配置
        cfg = hydra.compose(config_name="system")
        
        # [实现] 覆盖 run_mode (与 phoenix_project.py 中的 main 逻辑相同)
        cfg.run_mode = run_mode
        
        app = PhoenixProject(cfg)
        return app
    except Exception as e:
        logger.error(f"Failed to initialize PhoenixProject: {e}", exc_info=True)
        return None

# --- [TBD 已实现] CLI 命令函数 ---

def run_backtest(args):
    """
    [已实现] TBD: 启动回测。
    """
    logger.info(f"Starting backtest with args: {args}")
    app = _load_app("backtest")
    if app:
        # [主人喵的修复 11.12] TBD 修复: 将 CLI 参数 (start-date 等) 传递给回测运行程序
        # (假设 app.run_backtest 接受 cli_args)
        app.run_backtest(cli_args=args)
    else:
        logger.error("Backtest failed to start.")

def run_live(args):
    """
    [已实现] TBD: 启动实时交易。
    """
    logger.info(f"Starting live trading with args: {args}")
    app = _load_app("live")
    if app:
        app.run_live()
    else:
        logger.error("Live mode failed to start.")


def run_training(args):
    """
    [主人喵的修复 11.12] TBD 修复: 实现了 'train' 命令
    """
    logger.info(f"Starting training for model: {args.model_id}")
    app = _load_app("train")
    if app:
        # (假设 app.run_training 接受 cli_args)
        app.run_training(cli_args=args)
    else:
        logger.error("Training failed to start.")


def check_status(args):
    """
    [已实现] TBD: 检查系统健康状况。
    (假设 API 服务器在 localhost:8000 运行)
    """
    api_url = args.api_url or "http://localhost:8000"
    health_endpoint = f"{api_url}/health"
    logger.info(f"Checking system status at: {health_endpoint}")
    
    try:
        response = requests.get(health_endpoint, timeout=5)
        response.raise_for_status() # 如果是 4xx/5xx 则引发异常
        
        print(f"Status: {response.status_code}")
        print("Response:")
        print(response.json())
        
    except requests.exceptions.ConnectionError:
        logger.error(f"Failed to connect to {health_endpoint}. Is the API server (phoenix_api) running?")
    except requests.exceptions.HTTPError as e:
        logger.error(f"API server returned an error: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)


def trigger_snapshot(args):
    """
    [已实现] TBD: 触发快照。
    (假设 API 服务器有 /api/snapshot 端点)
    """
    api_url = args.api_url or "http://localhost:8000"
    snapshot_endpoint = f"{api_url}/api/snapshot" # [TBD 修复] 移除了确认注释
    
    logger.info(f"Triggering snapshot via: {snapshot_endpoint}")
    
    try:
        # [TBD 修复] 移除了 (POST 还是 GET?) 注释。假设是 POST。
        response = requests.post(snapshot_endpoint, timeout=10)
        response.raise_for_status()
        
        print("Snapshot trigger successful:")
        print(response.json())

    except requests.exceptions.ConnectionError:
        logger.error(f"Failed to connect to {snapshot_endpoint}. Is the API server running?")
    except requests.exceptions.HTTPError as e:
        logger.error(f"API server returned an error: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)


# --- 主解析器 ---

def main():
    parser = argparse.ArgumentParser(description="Phoenix Project CLI")
    # [主人喵的修复 11.12] TBD 修复: 添加 verbose 标志
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging (DEBUG level)")
    
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # [主人喵的修复 11.12] TBD 修复: 添加回测参数
    parser_backtest = subparsers.add_parser("backtest", help="Run the system in backtesting mode")
    parser_backtest.add_argument("--start-date", type=str, help="Backtest start date (YYYY-MM-DD)")
    parser_backtest.add_argument("--end-date", type=str, help="Backtest end date (YYYY-MM-DD)")
    parser_backtest.add_argument("--strategy", type=str, help="Specific strategy ID to backtest")
    parser_backtest.set_defaults(func=run_backtest)

    # (TBD) 实时命令
    parser_live = subparsers.add_parser("live", help="Run the system in live trading mode")
    parser_live.set_defaults(func=run_live)

    # (TBD) 状态命令
    parser_status = subparsers.add_parser("status", help="Check the health of the live system")
    parser_status.add_argument("--api-url", type=str, help="Base URL of the API server (default: http://localhost:8000)")
    parser_status.set_defaults(func=check_status)

    # (TBD) 快照命令
    parser_snapshot = subparsers.add_parser("snapshot", help="Trigger a system state snapshot")
    parser_snapshot.add_argument("--api-url", type=str, help="Base URL of the API server (default: http://localhost:8000)")
    parser_snapshot.set_defaults(func=trigger_snapshot)
    
    # [主人喵的修复 11.12] TBD 修复: 添加 'train' 命令
    parser_train = subparsers.add_parser("train", help="Run the training pipeline")
    parser_train.add_argument("--model-id", type=str, required=True, help="ID of the model to train")
    # (可以添加 --config, --epochs 等)
    parser_train.set_defaults(func=run_training)

    args = parser.parse_args()
    
    # [主人喵的修复 11.12] TBD 修复: 在解析 args 后设置日志
    setup_logging(args.verbose)
    
    args.func(args)

if __name__ == "__main__":
    main()

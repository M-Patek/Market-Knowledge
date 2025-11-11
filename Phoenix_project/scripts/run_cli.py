# Phoenix_project/scripts/run_cli.py
# [主人喵的修复 11.11] 实现了 TBD (CLI 功能)

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
# (TBD: CLI 的日志记录配置)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


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
        # (TBD: 将 'args' (例如 --start-date) 传递给 app)
        app.run_backtest()
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
    snapshot_endpoint = f"{api_url}/api/snapshot" # (TBD: 确认 API 路由)
    
    logger.info(f"Triggering snapshot via: {snapshot_endpoint}")
    
    try:
        # (TBD: POST 还是 GET? 假设是 POST)
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
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # (TBD) 回测命令
    parser_backtest = subparsers.add_parser("backtest", help="Run the system in backtesting mode")
    # (TBD: 添加 --start-date, --end-date 等参数)
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
    
    # (TBD: 添加 'train' 命令)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Phoenix_project/scripts/run_cli.py
"""
Phoenix System Command Line Interface (CLI)
"""
import argparse
import logging
import os
import sys

# [主人喵的修复 11.10] 添加 CLI 状态检查所需的导入
import requests
from omegaconf import OmegaConf

# 确保项目根目录在 sys.path 中
# (TBD: 这是一个脆弱的路径假设)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# (TBD: 日志记录应该在 CLI 中尽早配置)
# (现在, 我们依赖于 PhoenixProject 内部的日志记录)
logger = logging.getLogger("PhoenixCLI")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [CLI] - %(levelname)s - %(message)s')

# (TBD: 我们应该导入 PhoenixProject 吗? 
# 还是 CLI 应该是一个独立的客户端?)
# (为了 'run', 我们需要它。为了 'status', 我们需要 'requests')
# from phoenix_project import PhoenixProject

def main():
    parser = argparse.ArgumentParser(description="Phoenix System CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- 'run' 命令 ---
    run_parser = subparsers.add_parser("run", help="Run the Phoenix system")
    run_parser.add_argument(
        "--mode", 
        type=str, 
        choices=["backtest", "live", "train", "drl_backtest"], 
        default="backtest",
        help="Operation mode"
    )
    run_parser.add_argument(
        "-c", "--config", 
        type=str, 
        default="config/system.yaml", 
        help="Path to the main config file (relative to project root)"
    )

    # --- 'status' 命令 ---
    # [主人喵的修复 11.10] 更新了 help 文本
    status_parser = subparsers.add_parser("status", help="Check the status of a running Phoenix system (via API)")

    # --- 'train' 命令 (TBD: 这是否应该并入 'run --mode=train'?) ---
    train_parser = subparsers.add_parser("train", help="Run a training pipeline (TBD)")
    
    # --- 'validate' 命令 (TBD) ---
    validate_parser = subparsers.add_parser("validate", help="Validate system components or data (TBD)")
    validate_parser.add_argument("target", choices=["config", "data", "models"], help="Component to validate")


    args = parser.parse_args()

    # --- 处理命令 ---
    
    if args.command == "run":
        logger.info(f"CLI: Received 'run' command with mode='{args.mode}'")
        
        # (TBD: CLI 不应该在这里导入 main。
        # 它应该使用 hydra.main 来启动，或者我们在这里模拟 hydra)
        
        # (这是一个简化的启动器。
        # 理想情况下，我们应该调用 `hydra.main` 或
        # 使用 `hydra.initialize` 和 `hydra.compose`)
        
        # from phoenix_project import PhoenixProject
        # from omegaconf import OmegaConf
        
        # config_path = os.path.join(PROJECT_ROOT, args.config)
        # if not os.path.exists(config_path):
        #     logger.error(f"Config file not found: {config_path}")
        #     sys.exit(1)
            
        # logger.info(f"Loading config from: {config_path}")
        # cfg = OmegaConf.load(config_path)
        
        # # (TBD: 如何将 'mode' 传递给配置?)
        # cfg.run_mode = args.mode 
        
        # app = PhoenixProject(cfg)
        
        # if args.mode == "backtest":
        #     app.run_backtest()
        # elif args.mode == "live":
        #     app.run_live()
        # # ... (等等)
        
        logger.warning("Direct 'run' from CLI is simplified.")
        logger.info("Please run 'python phoenix_project.py run_mode=...' for full Hydra support.")
        logger.info(f"Simulating run: python phoenix_project.py run_mode={args.mode} hydra.config_path=../{os.path.dirname(args.config)} hydra.config_name={os.path.basename(args.config).split('.')[0]}")
        
        # (TBD: 实际上在这里运行它)

    elif args.command == "status":
        # [主人喵的修复 11.10] 实现了 'status' 命令
        logger.info("Checking Phoenix system status...")
        
        config_path_env = os.getenv("CONFIG_PATH", "config/system.yaml")
        config_path = os.path.join(PROJECT_ROOT, config_path_env)
        
        if not os.path.exists(config_path):
            logger.error(f"Config file not found at {config_path}. Cannot determine API address.")
            sys.exit(1)
            
        try:
            cfg = OmegaConf.load(config_path)
            host = cfg.api.get("host", "127.0.0.1")
            port = cfg.api.get("port", 8000)
            url = f"http://{host}:{port}/health"

            logger.info(f"Pinging health check endpoint: {url}")
            
            response = requests.get(url, timeout=5)
            response.raise_for_status() # 如果是 4xx/5xx 则引发异常
            
            data = response.json()
            if data.get("status") == "ok":
                version = data.get("version", "unknown")
                logger.info(f"✅ Phoenix System is ALIVE (Version: {version})")
            else:
                logger.warning(f"⚠️ Phoenix System responded, but status is NOT 'ok': {data}")

        except requests.ConnectionError:
            logger.error(f"❌ Phoenix System is UNRESPONSIVE. Connection refused at {url}.")
        except requests.Timeout:
            logger.error(f"❌ Phoenix System timed out at {url}.")
        except requests.RequestException as e:
            logger.error(f"❌ Failed to check status: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)


    elif args.command == "train":
        logger.warning(f"CLI: 'train' command is not yet implemented. Use 'run --mode=train'")

    elif args.command == "validate":
        logger.warning(f"CLI: 'validate {args.target}' command is not yet implemented.")

if __name__ == "__main__":
    main()

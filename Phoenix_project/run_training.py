"""
训练执行脚本
用于离线训练和验证 AI/DRL 模型。
"""

import argparse
import json
import os
from typing import Dict, Any

from Phoenix_project.config.loader import ConfigLoader
from Phoenix_project.data_manager import DataManager
from Phoenix_project.monitor.logging import setup_logging, get_logger

# (根据需要导入具体的训练器)
from Phoenix_project.training.walk_forward_trainer import WalkForwardTrainer
from Phoenix_project.training.drl.multi_agent_trainer import MultiAgentTrainer
from Phoenix_project.training.engine import BacktestingEngine # 用于评估
from Phoenix_project.training.drl.trading_env import TradingEnv # DRL 环境

def load_data_catalog(config_loader: ConfigLoader) -> Dict[str, Any]:
    """
    加载数据目录。
    """
    catalog_path = config_loader.get_system_config().get("data_catalog_path", "data_catalog.json")
    if not os.path.exists(catalog_path):
        get_logger(__name__).error(f"Data catalog not found at {catalog_path}")
        return {}
    try:
        with open(catalog_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        get_logger(__name__).error(f"Failed to load data catalog: {e}", exc_info=True)
        return {}

def main():
    parser = argparse.ArgumentParser(description="Phoenix Project - Model Training")
    parser.add_argument(
        '--config_path', 
        type=str, 
        default='config', 
        help='Path to the configuration directory.'
    )
    parser.add_argument(
        '--trainer', 
        type=str, 
        required=True, 
        choices=['walk_forward', 'drl_multi_agent'], 
        help='The type of trainer to run.'
    )
    # (可以添加更多参数，如 start_date, end_date, symbols)
    
    args = parser.parse_args()
    
    setup_logging()
    logger = get_logger(__name__)
    logger.info(f"Starting training process: {args.trainer}")

    try:
        # 1. 加载配置和数据
        config_loader = ConfigLoader(args.config_path)
        data_catalog = load_data_catalog(config_loader)
        
        # FIX (E5): DataManager 构造函数需要 ConfigLoader，而不是 dict
        data_manager = DataManager(config_loader, data_catalog)
        
        # 2. 初始化训练器
        if args.trainer == 'walk_forward':
            # (示例：初始化 WalkForwardTrainer)
            backtest_engine = BacktestingEngine(data_manager, config_loader)
            trainer = WalkForwardTrainer(
                engine=backtest_engine,
                config_loader=config_loader
            )
            # (需要设置参数)
            # trainer.run_optimization(...)

        elif args.trainer == 'drl_multi_agent':
            # (示例：初始化 DRL Trainer)
            
            # 1. 创建 DRL 环境
            # (需要从 DataManager 加载数据并传入)
            # df_market = data_manager.get_market_data(...)
            env_config = config_loader.get_system_config().get("drl_env", {})
            trading_env = TradingEnv(
                # market_data=df_market, 
                **env_config
            )
            
            # 2. 创建训练器
            trainer_config = config_loader.get_system_config().get("drl_trainer", {})
            trainer = MultiAgentTrainer(
                env=trading_env,
                **trainer_config
            )
            
            logger.info("Starting DRL Multi-Agent training...")
            trainer.train()
            logger.info("DRL training complete.")

        else:
            logger.error(f"Unknown trainer type: {args.trainer}")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)

if __name__ == "__main__":
    main()

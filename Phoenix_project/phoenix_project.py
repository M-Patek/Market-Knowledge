import asyncio
import os
import argparse
from typing import Dict, Any
from datetime import datetime
import json # [任务 C.4] 导入 json

# --- 核心组件 ---
from Phoenix_project.config.loader import ConfigLoader
from Phoenix_project.monitor.logging import get_logger
from Phoenix_project.data_manager import DataManager
from Phoenix_project.data.data_iterator import DataIterator
from Phoenix_project.controller.orchestrator import Orchestrator
from Phoenix_project.cognitive.engine import CognitiveEngine
from Phoenix_project.execution.trade_lifecycle_manager import TradeLifecycleManager
from Phoenix_project.output.renderer import render_report
from Phoenix_project.training.engine import BacktestingEngine

# --- AI/L3 组件 (用于 DRL 回测) ---
from Phoenix_project.features.store import FeatureStore
from Phoenix_project.agents.l3.alpha_agent import AlphaAgent
from Phoenix_project.agents.l3.risk_agent import RiskAgent
from Phoenix_project.agents.l3.execution_agent import ExecutionAgent

# (TBD: 导入其他组件，如 API 服务器, KG 服务等)

logger = get_logger("PhoenixProject")

class PhoenixProject:
    """
    Main application class for the Phoenix Project.
    Initializes and wires together all core components.
    """

    def __init__(self, config_path: str, run_mode: str = "backtest"):
        """
        Initializes the entire system.
        
        Args:
            config_path: Path to the main configuration directory.
            run_mode: "backtest" or "live".
        """
        logger.info(f"--- Initializing Phoenix Project (Mode: {run_mode}) ---")
        
        # 1. 配置
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.load_config('system.yaml')
        self.data_catalog = self.load_data_catalog()
        
        # 2. 核心执行与状态
        self.trade_lifecycle_manager = TradeLifecycleManager(self.config)
        self.data_manager = DataManager(self.config_loader, self.data_catalog)
        self.data_iterator = DataIterator(self.config, self.data_manager)
        
        # 3. 认知 (AI 核心)
        self.cognitive_engine = CognitiveEngine(self.config, self.data_manager)
        
        # 4. 编排器 (大脑)
        self.orchestrator = Orchestrator(
            config=self.config,
            data_manager=self.data_manager,
            cognitive_engine=self.cognitive_engine,
            trade_lifecycle_manager=self.trade_lifecycle_manager
        )
        
        # 5. 回测/训练引擎
        self.backtest_engine = BacktestingEngine(
            config=self.config,
            data_iterator=self.data_iterator,
            orchestrator=self.orchestrator,
            trade_lifecycle_manager=self.trade_lifecycle_manager,
            report_renderer_func=render_report # [FIX-15]
        )
        
        # 6. DRL/L3 组件 (仅在 DRL 回测时需要)
        # (延迟加载)
        self.feature_store: Optional[FeatureStore] = None
        self.l3_agents: Dict[str, Any] = {}

        logger.info("--- Phoenix Project Initialized Successfully ---")

    def load_data_catalog(self) -> Dict[str, Any]:
        """ Helper to load the data catalog JSON. """
        catalog_path = self.config.get("data_catalog_path", "data_catalog.json")
        try:
            # TBD: Use ConfigLoader's path logic
            full_path = os.path.join(self.config_loader.config_path, catalog_path)
            with open(full_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load data catalog from {catalog_path}: {e}", exc_info=True)
            return {}

    async def run_event_backtest(self):
        """
        Runs a full, event-driven backtest (L1/L2 Cognitive).
        """
        logger.info("--- Starting L1/L2 Cognitive Backtest ---")
        await self.backtest_engine.run()
        logger.info("--- L1/L2 Cognitive Backtest Complete ---")
        
    def run_drl_backtest(self):
        """
        Runs a DRL (L3) backtest.
        
        [任务 C.4] TODO: Add logic to print backtest results.
        """
        logger.info("--- Starting L3 DRL Backtest ---")
        
        # 1. (延迟) 加载 DRL 组件
        if not self.feature_store:
            fs_base_path = self.config.get("data_store", {}).get("local_base_path", "data")
            self.feature_store = FeatureStore(base_path=fs_base_path)
            
        if not self.l3_agents:
            # (TBD: 从 system.yaml 加载已训练模型的路径)
            logger.warning("Loading DRL L3 agents with MOCK paths. Update system.yaml.")
            self.l3_agents = {
                'alpha': AlphaAgent(model_path="models/drl/alpha_agent_checkpoint"),
                'risk': RiskAgent(model_path="models/drl/risk_agent_checkpoint"),
                'exec': ExecutionAgent(model_path="models/drl/exec_agent_checkpoint")
            }
            
        # 2. (TBD: 从 config 获取参数)
        bt_params = {
            "symbol": "AAPL",
            "start_date": datetime(2023, 1, 1),
            "end_date": datetime(2024, 1, 1),
        }

        # 3. 运行 DRL 回测 (任务 A.2)
        try:
            results = self.backtest_engine.run_backtest(
                data_manager=self.data_manager,
                feature_store=self.feature_store,
                alpha_agent=self.l3_agents['alpha'],
                risk_agent=self.l3_agents['risk'],
                execution_agent=self.l3_agents['exec'],
                **bt_params
            )
            
            # 4. [任务 C.4] 打印 DRL 回测结果
            if results:
                logger.info("--- DRL Backtest Results ---")
                try:
                    # 使用 json.dumps 美化输出
                    results_str = json.dumps(results, indent=2)
                    logger.info(results_str)
                except Exception as e:
                    logger.error(f"Could not serialize backtest results: {e}")
                    logger.info(results) # 回退到
            else:
                logger.warning("DRL Backtest returned no results.")

        except Exception as e:
            logger.critical(f"DRL Backtest failed to run: {e}", exc_info=True)
            logger.critical("Did you run DRL training (run_training.py) first?")
            
        logger.info("--- L3 DRL Backtest Complete ---")

    async def run_live(self):
        """
        Starts the system in live trading mode.
        """
        logger.info("--- Starting Phoenix Project (Live Mode) ---")
        # TBD:
        # 1. Start API server
        # 2. Start Orchestrator loop (e.g., orchestrator.start_live_processing())
        # 3. Connect to live data streams
        logger.warning("Live trading mode is not fully implemented.")
        # Example: await self.orchestrator.start_live_processing()
        pass

# --- 命令行入口 ---

async def main():
    parser = argparse.ArgumentParser(description="Phoenix Project Trading System")
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["backtest", "backtest-drl", "live"],
        default="backtest",
        help="The operational mode (default: backtest)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config",
        help="Path to the configuration directory (default: config/)."
    )
    
    args = parser.parse_args()
    
    # (TBD: 设置环境变量, e.g., os.environ['PHOENIX_CONFIG_PATH'] = args.config)
    
    try:
        app = PhoenixProject(config_path=args.config, run_mode=args.mode)
        
        if args.mode == "backtest":
            await app.run_event_backtest()
        elif args.mode == "backtest-drl":
            app.run_drl_backtest()
        elif args.mode == "live":
            await app.run_live()
            
    except FileNotFoundError as e:
        logger.critical(f"Configuration file not found. Path: {args.config}. Error: {e}")
    except Exception as e:
        logger.critical(f"Phoenix Project failed to run: {e}", exc_info=True)

if __name__ == "__main__":
    # (TBD: Add uvloop for performance if desired)
    asyncio.run(main())

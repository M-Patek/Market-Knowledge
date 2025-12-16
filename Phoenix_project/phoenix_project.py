"""
Phoenix Project (Market Knowledge) - 主入口点
[Code Opt Expert Fix] Task 18: Pre-Trade Risk Manager Warm-up
[Fix IV.1] Registry & API Key Fixes
[Task 005] Fix Startup Race Condition (Order: Ledger -> Risk -> API -> Loop)
[P1-INFRA-01] Explicit Shutdown & Lifecycle Refactor
"""

import asyncio
import logging
import os
import signal
import threading
from typing import Dict, Any, Optional
import hydra
import redis.asyncio as redis
from omegaconf import DictConfig, OmegaConf

from Phoenix_project.monitor.logging import setup_logging
from Phoenix_project.api.gemini_pool_manager import GeminiPoolManager 
from Phoenix_project.ai.gemini_search_adapter import GeminiSearchAdapter
from Phoenix_project.memory.vector_store import VectorStore
from Phoenix_project.memory.cot_database import CoTDatabase
from Phoenix_project.context_bus import ContextBus
from Phoenix_project.factory import PhoenixFactory # [Task 8] Import Factory
from Phoenix_project.ai.graph_db_client import GraphDBClient
from Phoenix_project.ai.tabular_db_client import TabularDBClient
from Phoenix_project.ai.temporal_db_client import TemporalDBClient
from Phoenix_project.ai.gnn_inferencer import GNNInferencer
from Phoenix_project.ai.retriever import Retriever
from Phoenix_project.ai.data_adapter import DataAdapter
from Phoenix_project.ai.relation_extractor import RelationExtractor
from Phoenix_project.ai.prompt_manager import PromptManager
from Phoenix_project.ai.prompt_renderer import PromptRenderer
from Phoenix_project.ai.ensemble_client import EnsembleClient
from Phoenix_project.ai.embedding_client import EmbeddingClient
from Phoenix_project.agents.executor import AgentExecutor
from Phoenix_project.agents.l3.alpha_agent import AlphaAgent
from Phoenix_project.agents.l3.risk_agent import RiskAgent
from Phoenix_project.agents.l3.execution_agent import ExecutionAgent
from Phoenix_project.registry import Registry 
from Phoenix_project.agents.l2.fusion_agent import FusionAgent
from Phoenix_project.evaluation.fact_checker import FactChecker
from Phoenix_project.evaluation.arbitrator import Arbitrator
from Phoenix_project.evaluation.voter import Voter
from Phoenix_project.cognitive.engine import CognitiveEngine
from Phoenix_project.cognitive.risk_manager import RiskManager
from Phoenix_project.cognitive.portfolio_constructor import PortfolioConstructor
from Phoenix_project.execution.order_manager import OrderManager
from Phoenix_project.execution.trade_lifecycle_manager import TradeLifecycleManager
from Phoenix_project.execution.adapters import AlpacaAdapter
from Phoenix_project.execution.interfaces import IBrokerAdapter
from Phoenix_project.events.event_distributor import EventDistributor
from Phoenix_project.events.risk_filter import EventRiskFilter
from Phoenix_project.ai.reasoning_ensemble import ReasoningEnsemble
from Phoenix_project.ai.market_state_predictor import MarketStatePredictor
from Phoenix_project.controller.orchestrator import Orchestrator
from Phoenix_project.controller.loop_manager import LoopManager
from Phoenix_project.data_manager import DataManager
from Phoenix_project.knowledge_injector import KnowledgeInjector
from Phoenix_project.interfaces.api_server import APIServer
from Phoenix_project.audit_manager import AuditManager

# 设置日志 (在最顶部)
setup_logging()
logger = logging.getLogger(__name__)

class PhoenixProject:
    """
    Phoenix Project 主应用类，负责服务初始化和生命周期管理。
    """
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        logger.info(f"Initializing PhoenixProject with run_mode: {cfg.get('run_mode', 'default')}")
        self.services = self._create_services(cfg)
        self.systems = self._create_main_systems(self.services)

    def _create_services(self, config: DictConfig) -> Dict[str, Any]:
        logger.info("Initializing core services via Registry...")
        
        # [Task 3.2] Activate Agent Loading via Registry
        # 1. Initialize Registry (Builds Core Infrastructure)
        self.registry = Registry(config)
        
        # 2. Build System (Agents, Engines, Orchestrator)
        sys_ctx = self.registry.build_system(config)
        
        # [Task FIX-MED-001] Removed manual L3/Execution instantiation.
        # Everything is now wired inside Registry.build_system().

        logger.info("Core services initialized and wired.")
        
        # Return dictionary matching expected structure + Registry context
        return {
            "config": config,
            "registry": self.registry,
            "sys_ctx": sys_ctx, # Pass full context
            
            # Map specific keys expected by legacy methods
            "context_bus": sys_ctx.context_bus,
            "redis_client": self.registry.container.data_manager.redis_client,
            "data_manager": sys_ctx.data_manager,
            "gemini_manager": self.registry.container.gemini_manager,
            
            # Clients
            "ensemble_client": self.registry.container.ensemble_client,
            "embedding_client": self.registry.container.embedding_client,
            
            # DBs
            "vector_store": self.registry.container.vector_store,
            "graph_db": self.registry.container.graph_db_client,
            "tabular_db": self.registry.container.tabular_db,
            "cot_db": self.registry.container.cot_database,
            
            # Engines & Managers
            "agent_executor": getattr(sys_ctx, 'agent_executor', None),
            "cognitive_engine": sys_ctx.cognitive_engine,
            "risk_manager": sys_ctx.risk_manager,
            "order_manager": sys_ctx.order_manager,
            # [Task FIX-MED-001] Retrieve from sys_ctx
            "trade_lifecycle_manager": sys_ctx.trade_lifecycle_manager,
            "broker_adapter": sys_ctx.broker_adapter,
            "audit_manager": sys_ctx.audit_manager,
            
            # Event components
            "event_distributor": sys_ctx.event_distributor,
            "event_filter": getattr(sys_ctx, 'event_filter', None), 
            
            # Agents (Optional mapping)
            "alpha_agent": sys_ctx.l3_agents.get("alpha"),
            "risk_agent": sys_ctx.l3_agents.get("risk"),
            "execution_agent": sys_ctx.l3_agents.get("execution"),

            # [Fix Attribute Error] Explicitly provide data_adapter
            "data_adapter": self.registry.container.data_adapter,
        }

    def _create_main_systems(self, services: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Initializing main systems...")
        
        # [Refactor] Fail-fast retrieval of Orchestrator
        # P1-INFRA-01: Remove dead fallback code and assertion logic
        sys_ctx = services.get("sys_ctx")
        if not sys_ctx or not getattr(sys_ctx, "orchestrator", None):
            raise RuntimeError("Critical: Orchestrator not found in Registry context.")
            
        orchestrator = sys_ctx.orchestrator
        audit_manager = services["audit_manager"]

        # (L0/L1/L2) 循环管理器 (Loop Manager)
        # [Task P1-DATA-02] Drift correction enabled implicitly via LoopManager update
        loop_manager = LoopManager(
            interval=services["config"].controller.get("loop_interval", 1.0),
            bus=services["context_bus"]
        )
    
        # (RAG) 知识注入器 (Knowledge Injector)
        knowledge_injector = KnowledgeInjector(
            vector_store=services["vector_store"],
            graph_db=services["graph_db"],
            # [Fix] Use data_adapter from services directly
            data_adapter=services.get("data_adapter"),
            relation_extractor=services.get("relation_extractor") or getattr(sys_ctx, "knowledge_graph_service", None).relation_extractor,
            data_manager=services["data_manager"]
        )

        logger.info("Main systems initialized.")
        return {
            "orchestrator": orchestrator,
            "loop_manager": loop_manager,
            "data_manager": services["data_manager"],
            "knowledge_injector": knowledge_injector,
            "audit_manager": audit_manager,
            "trade_lifecycle_manager": services["trade_lifecycle_manager"]
        }

    async def shutdown(self):
        """
        [P1-INFRA-01] Explicit Graceful Shutdown Sequence.
        Ensures all components are stopped in the correct order:
        Loop -> API -> Data Connections.
        """
        logger.info("Initiating graceful shutdown sequence...")
        
        # 1. Stop Loop Manager (Stop accepting new events/tasks)
        if self.systems.get("loop_manager"):
            logger.info("Stopping Loop Manager...")
            # Calls stop (was previously stop_loop)
            await self.systems["loop_manager"].stop()

        # 2. Stop API Server
        if hasattr(self, 'api_server') and self.api_server:
            logger.info("Stopping API Server...")
            if hasattr(self.api_server, "stop"):
                server_stop = self.api_server.stop
                if asyncio.iscoroutinefunction(server_stop):
                    await server_stop()
                else:
                    server_stop()
            else:
                logger.warning("API Server does not have a stop() method.")

        # 3. Close Data Resources
        if hasattr(self, 'services'):
            logger.info("Closing Data Resources...")
            
            async def _close_resource(name: str, resource: Any):
                if resource and hasattr(resource, "close"):
                    try:
                        c = resource.close
                        if asyncio.iscoroutinefunction(c):
                            await c()
                        else:
                            c()
                        logger.info(f"{name} closed.")
                    except Exception as e:
                        logger.error(f"Error closing {name}: {e}")

            # Close critical DB connections in parallel or sequence
            await asyncio.gather(
                _close_resource("DataManager", self.services.get("data_manager")),
                _close_resource("GraphDB", self.services.get("graph_db")),
                _close_resource("TabularDB", self.services.get("tabular_db"))
            )

        logger.info("Shutdown sequence complete.")

    def run_backtest(self, cli_args):
        """
        [CLI] 运行回测模式
        """
        logger.info(f"Starting Backtest with args: {cli_args}")
        # TODO: Implement generic backtest execution logic here or in a separate engine
        pass

    def run_training(self, cli_args):
        """
        [CLI] 运行训练模式
        """
        logger.info(f"Starting Training with args: {cli_args}")
        pass

    def run_live(self):
        """
        [CLI/Main] 启动实时交易循环 (同步入口，内部运行异步循环)
        """
        try:
            asyncio.run(self._async_run_live())
        except KeyboardInterrupt:
            # Main loop cancelled by user (if not caught inside)
            logger.info("Phoenix Project terminated by user.")

    async def _async_run_live(self):
        """
        系统主异步循环逻辑。
        """
        logger.info("--- Phoenix Project Starting (Live Mode) ---")
        
        try:
            # --- 3. 获取核心系统 ---
            loop_manager = self.systems["loop_manager"]
            orchestrator = self.systems["orchestrator"]
            trade_lifecycle_manager = self.systems["trade_lifecycle_manager"]
            risk_manager = self.services["risk_manager"]
            
            # --- 4. 关键依赖初始化 (Warm-up Phase) ---
            logger.info("Starting system warm-up...")
            
            # [Task 4.1] 初始化持久化账本 (必须在任何交易逻辑或 API 暴露之前)
            # 这确保我们知道当前实际持仓
            await trade_lifecycle_manager.initialize()
            logger.info("Trade Lifecycle Manager (Ledger) initialized.")

            # [Task 18] Risk Manager 初始化 (Pre-Trade Warm-up)
            # 这确保风险检查器已加载历史数据和规则
            symbols = [self.cfg.trading.get('default_symbol', 'BTC/USD')]
            await risk_manager.initialize(symbols)
            logger.info("Risk Manager initialized and warmed up.")
            
            # --- 5. 启动外部接口 (API Server) ---
            # [Task 005] 只有在内部状态 (Ledger & Risk) 就绪后，才暴露 API
            # [Task FIX-CRIT-003] Pass main_loop to APIServer for thread-safe scheduling
            main_loop = asyncio.get_running_loop()
            
            self.api_server = APIServer(
                host=self.cfg.api_gateway.host,
                port=self.cfg.api_gateway.port,
                context_bus=self.services["context_bus"],
                logger=logger,
                audit_viewer=None, # TBD
                main_loop=main_loop
            )

            # [Task 5.1 Fix] Offload API Server to Daemon Thread
            # Note: APIServer.run spawns its own thread, so we can call it directly or wrap it.
            # Given current impl of APIServer.run starts a thread, we can just call it.
            self.api_server.run() 
            logger.info("API Server started (Background Uvicorn).")

            # --- 6. 启动主循环 (Loop Manager) ---
            logger.info("Starting Orchestrator Loop...")
            
            # [Refactor] Use start() instead of start_loop() to match new LoopManager interface
            # Passing orchestrator.run_cycle as the work function
            await loop_manager.start(orchestrator.run_cycle)
            
            # [Phase IV Fix] Graceful Shutdown Handling
            loop = asyncio.get_running_loop()
            stop_event = asyncio.Event()
            
            def _signal_handler():
                logger.info("Shutdown signal received. Initiating graceful stop...")
                stop_event.set()
                # loop_manager.stop() is async, handled in shutdown sequence or we can trigger event here
            
            # Register signal handlers
            for sig in (signal.SIGTERM, signal.SIGINT):
                try:
                    loop.add_signal_handler(sig, _signal_handler)
                except NotImplementedError:
                    logger.warning(f"Signal handling for {sig} not supported on this platform.")

            # Create a task that waits for the stop signal
            logger.info("System running. Press Ctrl+C to stop.")
            await stop_event.wait()
            
            logger.info("--- Phoenix Project Shutting Down ---")
            
        except asyncio.CancelledError:
            logger.info("Main loop cancelled.")
        except Exception as e:
            logger.critical(f"Fatal error in main: {e}", exc_info=True)

        finally:
            # [P1-INFRA-01] Ensure explicit cleanup via shutdown()
            # Removed magic asyncio.sleep(1)
            await self.shutdown()
            logger.info("Phoenix Project exit.")


@hydra.main(version_base=None, config_path="config", config_name="system")
def main(cfg: DictConfig):
    """
    主入口点：由 Hydra 处理配置加载。
    """
    # 1. 初始化应用
    app = PhoenixProject(cfg)
    # 2. 运行 (默认 Live 模式)
    app.run_live()

if __name__ == "__main__":
    main()

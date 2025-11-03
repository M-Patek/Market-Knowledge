"""
凤凰计划 (Phoenix Project) 主应用程序入口点。
负责初始化和协调所有核心组件。
"""
import json
import os
# FIX (E-PY-1): 从 'typing' 中导入 'List'，以修复 run_backtest 中的 NameError
from typing import Dict, Any, List

from Phoenix_project.config.loader import ConfigLoader
from Phoenix_project.data_manager import DataManager
from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.events.event_distributor import EventDistributor
from Phoenix_project.events.risk_filter import EventRiskFilter
from Phoenix_project.events.stream_processor import StreamProcessor
from Phoenix_project.cognitive.engine import CognitiveEngine
from Phoenix_project.cognitive.portfolio_constructor import PortfolioConstructor
from Phoenix_project.cognitive.risk_manager import RiskManager
from Phoenix_project.execution.order_manager import OrderManager
from Phoenix_project.execution.trade_lifecycle_manager import TradeLifecycleManager

# FIX (E6): 导入 AlpacaAdapter (我们将在 adapters.py 中添加它)
from Phoenix_project.execution.adapters import SimulatedBrokerAdapter, AlpacaAdapter
from Phoenix_project.controller.orchestrator import Orchestrator
from Phoenix_project.controller.loop_manager import LoopManager
from Phoenix_project.controller.error_handler import ErrorHandler
from Phoenix_project.controller.scheduler import Scheduler
from Phoenix_project.audit_manager import AuditManager
from Phoenix_project.snapshot_manager import SnapshotManager
from Phoenix_project.metrics_collector import MetricsCollector
from Phoenix_project.monitor.logging import setup_logging, get_logger

# (AI/RAG components)
from Phoenix_project.ai.retriever import Retriever
from Phoenix_project.ai.ensemble_client import EnsembleClient
from Phoenix_project.ai.metacognitive_agent import MetacognitiveAgent
from Phoenix_project.ai.reasoning_ensemble import ReasoningEnsemble
from Phoenix_project.evaluation.arbitrator import Arbitrator
from Phoenix_project.evaluation.fact_checker import FactChecker
# FIX: Add missing imports for CognitiveEngine dependencies
from Phoenix_project.evaluation.voter import Voter
from Phoenix_project.fusion.uncertainty_guard import UncertaintyGuard
from Phoenix_project.ai.prompt_manager import PromptManager
from Phoenix_project.api.gateway import APIGateway
from Phoenix_project.api.gemini_pool_manager import GeminiPoolManager
from Phoenix_project.memory.vector_store import VectorStore
from Phoenix_project.memory.cot_database import CoTDatabase
from Phoenix_project.sizing.fixed_fraction import FixedFractionSizer # 示例仓位管理器

class PhoenixProject:
    """
    主应用程序类。
    """
    def __init__(self, config_path: str = 'config'):
        # 1. 配置与日志
        setup_logging()
        self.logger = get_logger(__name__)
        self.logger.info("Phoenix Project V2.0 启动中...")
        
        self.config_loader = ConfigLoader(config_path)
        
        # 2. 加载数据
        catalog_path = self.config_loader.get_system_config().get("data_catalog_path", "data_catalog.json")
        self.data_catalog = self._load_data_catalog(catalog_path)
        
        # 3. 初始化核心组件
        self.error_handler = ErrorHandler()
        
        # FIX (E5): DataManager 构造函数需要 ConfigLoader，而不是 dict
        self.data_manager = DataManager(self.config_loader, self.data_catalog)
        
        # FIX (E5): PipelineState 构造函数需要 initial_state 和 max_history
        max_history = self.config_loader.get_system_config().get("max_pipeline_history", 100)
        self.pipeline_state = PipelineState(initial_state=None, max_history=max_history)
        
        self.audit_manager = AuditManager(self.config_loader)
        self.snapshot_manager = SnapshotManager(self.config_loader)
        
        # 4. 初始化事件/数据流
        self.event_distributor = EventDistributor()
        self.event_filter = EventRiskFilter(self.config_loader)
        self.stream_processor = StreamProcessor()

        # 5. 初始化券商 (Execution)
        self.broker = self._setup_broker()
        
        # 6. 初始化交易生命周期
        initial_cash = self.config_loader.get_system_config().get("trading", {}).get("initial_cash", 1000000)
        self.trade_lifecycle_manager = TradeLifecycleManager(initial_cash=initial_cash)
        self.order_manager = OrderManager(self.broker)
        
        # 7. 初始化AI/认知 (Cognitive)
        self.cognitive_engine = self._setup_cognitive_engine()
        
        # 8. 初始化投资组合
        # (使用一个示例仓位管理器)
        sizer = FixedFractionSizer(fraction=0.05) # 示例：每次交易使用5%的资金
        self.portfolio_constructor = PortfolioConstructor(position_sizer=sizer)
        self.risk_manager = RiskManager(self.config_loader)

        # 9. 监控
        self.metrics_collector = MetricsCollector(self.config_loader)

        # 10. 核心协调器 (Orchestrator)
        self.orchestrator = Orchestrator(
            config_loader=self.config_loader, # FIX: Added missing config_loader
            pipeline_state=self.pipeline_state,
            data_manager=self.data_manager,
            event_distributor=self.event_distributor, # FIX: Pass event_distributor
            cognitive_engine=self.cognitive_engine,
            portfolio_constructor=self.portfolio_constructor,
            risk_manager=self.risk_manager,
            order_manager=self.order_manager,
            # FIX: Removed extraneous args (event_filter, stream_processor, trade_lifecycle_manager)
            snapshot_manager=self.snapshot_manager,
            metrics_collector=self.metrics_collector,
            audit_manager=self.audit_manager,
            error_handler=self.error_handler
        )

        # 11. 循环与调度
        self.loop_manager = LoopManager(self.orchestrator, self.data_manager)
        self.scheduler = Scheduler(self.orchestrator, self.config_loader)

        self.logger.info("Phoenix Project 初始化完成。")

    def _load_data_catalog(self, catalog_path: str) -> Dict[str, Any]:
        if not os.path.exists(catalog_path):
            self.logger.error(f"Data catalog not found at {catalog_path}")
            return {}
        try:
            with open(catalog_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load data catalog: {e}", exc_info=True)
            return {}

    def _setup_broker(self) -> SimulatedBrokerAdapter:
        # (FIX (E6): 暂时只支持模拟券商)
        # 
        # broker_config = self.config_loader.get_system_config().get("broker", {})
        # if broker_config.get("type") == "alpaca":
        #     return AlpacaAdapter(
        #         api_key=os.environ.get(broker_config.get("api_key_env")),
        #         api_secret=os.environ.get(broker_config.get("api_secret_env")),
        #         base_url=broker_config.get("base_url")
        #     )
        self.logger.warning("Broker type not specified or 'alpaca' not implemented, defaulting to SimulatedBrokerAdapter.")
        return SimulatedBrokerAdapter()

    def _setup_cognitive_engine(self) -> CognitiveEngine:
        self.logger.info("Initializing Cognitive Stack...")
        # (这部分逻辑在 worker.py 中更完整，这里只是一个示例)
        try:
            # 1. API & Prompts
            gemini_pool = GeminiPoolManager() # 假设默认
            api_gateway = APIGateway(gemini_pool)
            prompt_manager = PromptManager(self.config_loader.config_path)
            
            # 2. Memory
            vector_store = VectorStore()
            cot_db = CoTDatabase()
            
            # 3. RAG
            retriever = Retriever(vector_store, cot_db)
            
            # 4. Agents
            agent_registry = self.config_loader.get_agent_registry()
            ensemble_client = EnsembleClient(api_gateway, prompt_manager, agent_registry)
            metacognitive_agent = MetacognitiveAgent(api_gateway, prompt_manager)
            arbitrator = Arbitrator(api_gateway, prompt_manager)
            fact_checker = FactChecker(api_gateway, prompt_manager)
            
            # FIX: Instantiate missing dependencies for CognitiveEngine
            uncertainty_guard = UncertaintyGuard()
            voter = Voter() # Assuming no-arg constructor

            # 5. Ensemble
            reasoning_ensemble = ReasoningEnsemble(
                retriever=retriever,
                ensemble_client=ensemble_client,
                metacognitive_agent=metacognitive_agent,
                arbitrator=arbitrator,
                fact_checker=fact_checker
            )
            
            # 6. Cognitive Engine
            # FIX: Pass all required arguments to CognitiveEngine constructor
            cognitive_engine = CognitiveEngine(
                reasoning_ensemble=reasoning_ensemble,
                fact_checker=fact_checker,
                uncertainty_guard=uncertainty_guard,
                voter=voter,
                # Pass the relevant config section
                config=self.config_loader.get_config("cognitive_engine", {}) 
            )
            self.logger.info("Cognitive Stack Initialized.")
            return cognitive_engine
        except Exception as e:
            self.logger.error(f"Failed to initialize cognitive stack: {e}", exc_info=True)
            raise

    def run_backtest(self, start_date: str, end_date: str, symbols: List[str]):
        """
        运行回测。
        """
        self.logger.info(f"Running backtest from {start_date} to {end_date} for {symbols}")
        try:
            self.loop_manager.run_backtest(start_date, end_date, symbols)
            self.logger.info("Backtest complete.")
            # TODO: 打印回测结果
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}", exc_info=True)
            self.error_handler.handle_critical_error(e)

    def run_live(self):
        """
        以实时模式运行 (使用调度器)。
        """
        self.logger.info("Starting Phoenix Project in LIVE mode...")
        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            self.logger.info("Shutting down live mode...")
            self.scheduler.stop()
        except Exception as e:
            self.logger.error("Live mode failed:", exc_info=True)
            self.error_handler.handle_critical_error(e)
            self.scheduler.stop()

if __name__ == "__main__":
    # (示例运行)
    # 在真实部署中，这将通过 CLI (scripts/run_cli.py) 或 worker (worker.py) 启动
    
    app = PhoenixProject()
    
    # 示例: 运行回测
    app.run_backtest(
        start_date="2023-01-01T00:00:00Z",
        end_date="2023-03-01T00:00:00Z",
        symbols=["AAPL", "MSFT"]
    )

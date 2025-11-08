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

# [阶段 3] 导入所有适配器
from Phoenix_project.execution.interfaces import IBrokerAdapter # [修复] 导入接口
from Phoenix_project.execution.adapters import SimulatedBrokerAdapter, PaperTradingBrokerAdapter
from Phoenix_project.controller.orchestrator import Orchestrator
from Phoenix_project.controller.loop_manager import LoopManager
from Phoenix_project.controller.error_handler import ErrorHandler
from Phoenix_project.controller.scheduler import Scheduler
from Phoenix_project.audit_manager import AuditManager
from Phoenix_project.snapshot_manager import SnapshotManager
from Phoenix_project.metrics_collector import MetricsCollector
from Phoenix_project.monitor.logging import setup_logging, get_logger

# (AI/RAG components)
# [修复] 导入 worker.py 中使用的更完整的 AI 栈
from Phoenix_project.ai.retriever import Retriever
from Phoenix_project.ai.ensemble_client import EnsembleClient
from Phoenix_project.agents.l2.metacognitive_agent import MetacognitiveAgent # [修复]
from Phoenix_project.ai.reasoning_ensemble import ReasoningEnsemble
from Phoenix_project.evaluation.arbitrator import Arbitrator
from Phoenix_project.evaluation.fact_checker import FactChecker
from Phoenix_project.evaluation.voter import Voter
from Phoenix_project.fusion.uncertainty_guard import UncertaintyGuard
from Phoenix_project.ai.prompt_manager import PromptManager
from Phoenix_project.api.gateway import APIGateway
from Phoenix_project.api.gemini_pool_manager import GeminiPoolManager
from Phoenix_project.memory.vector_store import get_vector_store # [修复]
from Phoenix_project.memory.cot_database import CoTDatabase

# [修复] 导入 GNN/KG 组件
from Phoenix_project.ai.embedding_client import EmbeddingClient
from Phoenix_project.ai.tabular_db_client import TabularDBClient
from Phoenix_project.ai.temporal_db_client import TemporalDBClient
from Phoenix_project.ai.relation_extractor import RelationExtractor
from Phoenix_project.knowledge_graph_service import KnowledgeGraphService

# [修复] 导入动态加载 Sizer 所需的
import importlib
import pydash
from Phoenix_project.sizing.base import IPositionSizer
from Phoenix_project.sizing.fixed_fraction import FixedFractionSizer # 示例仓位管理器

# [修复] 导入 registry
try:
    from Phoenix_project.registry import registry
except ImportError:
    # (日志记录器可能尚未设置)
    print("Warning: Could not import 'registry' from 'Phoenix_project.registry'. Using empty dict.")
    registry = {} # 回退


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
        self.system_config = self.config_loader.load_config('system.yaml')
        
        # 2. 加载数据
        catalog_path = self.system_config.get("data_catalog_path", "data_catalog.json")
        self.data_catalog = self._load_data_catalog(catalog_path)
        
        # 3. 初始化核心组件
        self.error_handler = ErrorHandler(config=self.system_config.get("error_handler", {}))
        
        # FIX (E5): DataManager 构造函数需要 ConfigLoader
        self.data_manager = DataManager(self.config_loader, self.data_catalog)
        
        # FIX (E5): PipelineState 构造函数
        max_history = self.system_config.get("max_pipeline_history", 100)
        self.pipeline_state = PipelineState(initial_state=None, max_history=max_history)
        
        self.audit_manager = AuditManager(self.config_loader)
        self.snapshot_manager = SnapshotManager(self.config_loader)
        
        # 4. 初始化事件/数据流
        self.event_distributor = EventDistributor()
        self.event_filter = EventRiskFilter(self.config_loader)
        # [修复] StreamProcessor 可能不再需要，但保留以防万一
        self.stream_processor = StreamProcessor() 

        # 5. 初始化券商 (Execution)
        # [阶段 3 & 4] 依赖注入
        self.broker = self._setup_broker() # <--- 依赖注入
        
        # 6. 初始化交易生命周期
        initial_cash = self.system_config.get("trading", {}).get("initial_cash", 1000000)
        self.trade_lifecycle_manager = TradeLifecycleManager(initial_cash=initial_cash)
        
        # [阶段 4] 注入 TLM
        self.order_manager = OrderManager(self.broker, self.trade_lifecycle_manager)
        
        # 7. 初始化AI/认知 (Cognitive)
        self.cognitive_engine = self._setup_cognitive_engine()
        
        # 8. 初始化投资组合
        self.portfolio_constructor = self._setup_portfolio_constructor()
        self.risk_manager = RiskManager(self.config_loader)

        # 9. 监控
        self.metrics_collector = MetricsCollector(self.config_loader)

        # 10. 核心协调器 (Orchestrator)
        self.orchestrator = Orchestrator(
            # config_loader=self.config_loader, # [FIX] Orchestrator 不再需要 config_loader
            pipeline_state=self.pipeline_state,
            data_manager=self.data_manager,
            event_filter=self.event_filter, # [FIX] 传入 event_filter
            event_distributor=self.event_distributor, # FIX: Pass event_distributor
            cognitive_engine=self.cognitive_engine,
            portfolio_constructor=self.portfolio_constructor,
            risk_manager=self.risk_manager,
            order_manager=self.order_manager,
            # [阶段 4] 注入 TLM
            trade_lifecycle_manager=self.trade_lifecycle_manager,
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
        catalog_file_path = os.path.join(self.config_loader.config_path, catalog_path)
        if not os.path.exists(catalog_file_path):
            self.logger.error(f"Data catalog not found at {catalog_file_path}")
            return {}
        try:
            with open(catalog_file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load data catalog: {e}", exc_info=True)
            return {}

    def _setup_broker(self) -> IBrokerAdapter:
        """
        [阶段 3 实现]
        根据 config/system.yaml 中的 'system.environment' 来设置券商适配器。
        """
        # system_config = self.config_loader.get_system_config() # 已在 __init__ 中加载
        env = self.system_config.get("system", {}).get("environment", "development")
        broker_config = self.system_config.get("broker", {})
        
        if env == "production":
            self.logger.info(f"Production environment detected. Attempting to load PaperTradingBrokerAdapter.")
            if broker_config.get("type") == "alpaca":
                api_key_env = broker_config.get("api_key_env")
                api_secret_env = broker_config.get("api_secret_env")
                
                if not api_key_env or not api_secret_env:
                     self.logger.critical("Broker config 'api_key_env' or 'api_secret_env' missing in system.yaml")
                     raise ValueError("Missing broker API key env var names")

                api_key = os.environ.get(api_key_env)
                api_secret = os.environ.get(api_secret_env)
                
                if not api_key or not api_secret:
                    self.logger.critical(f"Environment variables {api_key_env} or {api_secret_env} not set.")
                    raise ValueError("Missing Alpaca API credentials in environment")
                    
                broker = PaperTradingBrokerAdapter(
                    api_key=api_key,
                    api_secret=api_secret,
                    paper_base_url=broker_config.get("base_url")
                )
                broker.connect() # 启动时连接
                return broker
            else:
                self.logger.critical("Production environment selected, but no valid 'alpaca' broker configured.")
                raise ValueError("Production environment requires a valid broker configuration.")
        
        # 默认为 "development"
        self.logger.info(f"Development environment detected. Loading SimulatedBrokerAdapter.")
        # [阶段 1 变更] SimBroker 不再需要 DataManager
        return SimulatedBrokerAdapter()

    def _setup_cognitive_engine(self) -> CognitiveEngine:
        """
        [修复] 使其与 worker.py 中的 AI 栈设置保持一致。
        """
        self.logger.info("Initializing Cognitive Stack...")
        try:
            # 1. API & Prompts
            gemini_pool = GeminiPoolManager()
            api_gateway = APIGateway(gemini_pool)
            prompt_manager = PromptManager(self.config_loader.config_path)
            
            # 2. AI/RAG 依赖
            embedding_client = EmbeddingClient(model_name=self.system_config.get("ai",{}).get("embedding_client",{}).get("model", "text-embedding-004"))
            tabular_client = TabularDBClient(config=self.system_config.get("tabular_db", {}))
            temporal_client = TemporalDBClient(config=self.system_config.get("temporal_db", {}))
            relation_extractor = RelationExtractor(api_gateway, prompt_manager)
            
            vector_store = get_vector_store(
                config=self.system_config.get("ai", {}).get("vector_database", {}),
                embedding_client=embedding_client,
                logger=self.logger
            )
            cot_db = CoTDatabase(config=self.system_config.get("cot_database", {}), logger=self.logger)
            knowledge_graph_service = KnowledgeGraphService(config=self.system_config.get("neo4j_db", {}))
            
            retriever = Retriever(
                config_loader=self.config_loader,
                vector_store=vector_store,
                cot_database=cot_db,
                embedding_client=embedding_client,
                knowledge_graph_service=knowledge_graph_service,
                temporal_client=temporal_client,
                tabular_client=tabular_client
            )
            
            # 4. Agents
            agent_registry_config = self.config_loader.get_agent_registry()
            ensemble_client = EnsembleClient(api_gateway, prompt_manager, agent_registry_config, registry) 
            metacognitive_agent = MetacognitiveAgent(api_gateway, prompt_manager)
            arbitrator = Arbitrator(api_gateway, prompt_manager)
            fact_checker = FactChecker(api_gateway, prompt_manager)
            
            # 5. Ensemble
            reasoning_ensemble = ReasoningEnsemble(
                retriever=retriever,
                ensemble_client=ensemble_client,
                metacognitive_agent=metacognitive_agent,
                arbitrator=arbitrator,
                fact_checker=fact_checker
            )
            
            # 6. Cognitive Engine
            uncertainty_guard = UncertaintyGuard()
            voter = Voter()

            cognitive_engine = CognitiveEngine(
                reasoning_ensemble=reasoning_ensemble,
                fact_checker=fact_checker,
                uncertainty_guard=uncertainty_guard,
                voter=voter,
                config=self.system_config.get("cognitive_engine", {}) 
            )
            self.logger.info("Cognitive Stack Initialized.")
            return cognitive_engine
        except Exception as e:
            self.logger.error(f"Failed to initialize cognitive stack: {e}", exc_info=True)
            raise

    def _setup_portfolio_constructor(self) -> PortfolioConstructor:
        """
        [修复] 使其与 worker.py 中的 Sizer 加载逻辑保持一致。
        """
        sizer_config = self.system_config.get("portfolio", {}).get("sizer", {})
        sizer_type_name = sizer_config.get("type", "FixedFractionSizer")
        sizer_params = sizer_config.get("params", {})
        sizer: IPositionSizer
        try:
            module_name = pydash.snake_case(sizer_type_name.replace("Sizer", ""))
            module = importlib.import_module(f"Phoenix_project.sizing.{module_name}")
            SizerClass = getattr(module, sizer_type_name)
            
            if sizer_type_name == "FixedFractionSizer" and not sizer_params:
                 sizer_params = {"fraction_per_position": 0.05}
            elif sizer_type_name == "VolatilityParitySizer" and not sizer_params:
                 sizer_params = {"volatility_period": 20}

            sizer = SizerClass(**sizer_params)
            self.logger.info(f"成功从配置加载 Sizer: {sizer_type_name} (Params: {sizer_params})")
        except (ImportError, AttributeError, TypeError) as e:
            self.logger.error(f"无法从配置加载 sizer '{sizer_type_name}' (Params: {sizer_params}): {e}。回退到 FixedFractionSizer。")
            sizer = FixedFractionSizer(fraction_per_position=0.05)
        
        return PortfolioConstructor(position_sizer=sizer)


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
            self.error_handler.handle_critical_error(e, self.pipeline_state) # [修复]

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
            self.error_handler.handle_critical_error(e, self.pipeline_state) # [修复]
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

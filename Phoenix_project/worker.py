"""
Celery Worker
负责异步执行 Orchestrator 任务。
[Refactored] 使用 PhoenixFactory 统一初始化逻辑。
[Fix] 废弃全局 Orchestrator 单例，确保每次任务在独立 Event Loop 中构建异步组件。
"""
import os
import asyncio
from celery import Celery
from celery.schedules import crontab
from celery.signals import worker_process_init
from prometheus_client import start_http_server

# (导入所有组件以进行初始化)
from Phoenix_project.config.loader import ConfigLoader
from Phoenix_project.data_manager import DataManager
from Phoenix_project.core.pipeline_state import PipelineState
# [Task 8] Import Factory
from Phoenix_project.factory import PhoenixFactory
from Phoenix_project.events.event_distributor import EventDistributor
from Phoenix_project.events.risk_filter import EventRiskFilter
from Phoenix_project.cognitive.engine import CognitiveEngine
from Phoenix_project.cognitive.portfolio_constructor import PortfolioConstructor
from Phoenix_project.cognitive.risk_manager import RiskManager
from Phoenix_project.execution.order_manager import OrderManager
from Phoenix_project.execution.trade_lifecycle_manager import TradeLifecycleManager

# [阶段 3] 导入所有适配器
from Phoenix_project.execution.interfaces import IBrokerAdapter
from Phoenix_project.execution.adapters import SimulatedBrokerAdapter, PaperTradingBrokerAdapter, AlpacaAdapter
from Phoenix_project.controller.orchestrator import Orchestrator
from Phoenix_project.controller.error_handler import ErrorHandler
from Phoenix_project.audit_manager import AuditManager
from Phoenix_project.snapshot_manager import SnapshotManager
from Phoenix_project.metrics_collector import MetricsCollector
from Phoenix_project.monitor.logging import setup_logging, get_logger

# (AI/RAG components)
from Phoenix_project.ai.retriever import Retriever
from Phoenix_project.ai.ensemble_client import EnsembleClient
from Phoenix_project.agents.l2.metacognitive_agent import MetacognitiveAgent 
from Phoenix_project.ai.reasoning_ensemble import ReasoningEnsemble
from Phoenix_project.evaluation.arbitrator import Arbitrator
from Phoenix_project.evaluation.fact_checker import FactChecker
from Phoenix_project.api.gateway import APIGateway
from Phoenix_project.memory.vector_store import VectorStore
from Phoenix_project.memory.cot_database import CoTDatabase
from Phoenix_project.audit.logger import AuditLogger
from Phoenix_project.ai.prompt_manager import PromptManager
from Phoenix_project.ai.prompt_renderer import PromptRenderer
from Phoenix_project.evaluation.voter import Voter
from Phoenix_project.ai.graph_db_client import GraphDBClient

# --- [主人喵的修复 1] 导入 GNN/KG 架构缺失的组件 ---
from Phoenix_project.ai.embedding_client import EmbeddingClient
from Phoenix_project.ai.tabular_db_client import TabularDBClient
from Phoenix_project.ai.temporal_db_client import TemporalDBClient
from Phoenix_project.ai.relation_extractor import RelationExtractor
from Phoenix_project.knowledge_graph_service import KnowledgeGraphService
from Phoenix_project.ai.gnn_inferencer import GNNInferencer 
# --- [修复结束] ---

import json
import importlib
from Phoenix_project.sizing.base import IPositionSizer
from Phoenix_project.sizing.fixed_fraction import FixedFractionSizer 

# [主人喵的清洁计划 3] 导入 Janitor 任务
try:
    from Phoenix_project.scripts.run_system_janitor import run_all_cleanup_tasks
except ImportError:
    setup_logging()
    logger = get_logger(__name__)
    logger.critical("无法导入 'run_all_cleanup_tasks'。Janitor 任务将无法运行。")
    run_all_cleanup_tasks = None

# [Task III.1] 导入 GNN 训练流水线
try:
    from Phoenix_project.training.gnn.gnn_engine import run_gnn_training_pipeline
except ImportError:
    setup_logging()
    logger = get_logger(__name__)
    logger.critical("无法导入 'run_gnn_training_pipeline'。GNN 任务将无法运行。")
    run_gnn_training_pipeline = None

# [蓝图 2 修复] 导入 registry
try:
    from Phoenix_project.registry import Registry
    global_registry = Registry()
except ImportError:
    setup_logging() 
    logger = get_logger(__name__)
    logger.warning("Could not import 'Registry'. Using fallback.")
    global_registry = None


# --- Celery App Setup ---
celery_app = Celery(
    'phoenix_worker',
    broker=os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0'),
    backend=os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
)
celery_app.conf.update(
    task_track_started=True,
    broker_connection_retry_on_startup=True
)

# --- [Phase 0 Fix] 废弃全局单例，防止跨 Event Loop 污染 ---
# orchestrator_instance: Orchestrator = None  <-- REMOVED

# --- 蓝图 1：为 Worker 启动 Prometheus 服务器 ---
@worker_process_init.connect
def start_prometheus_server(**kwargs):
    port = int(os.environ.get('WORKER_METRICS_PORT', 8001))
    try:
        start_http_server(port)
        print(f"Prometheus metrics server for worker started on port {port}")
    except Exception as e:
        print(f"Failed to start Prometheus metrics server on port {port}: {e}")

@worker_process_init.connect
def preload_gnn_model(**kwargs):
    """
    [Task 5A] Pre-load the GNN model as a Singleton when the worker process starts.
    """
    logger = get_logger(__name__)
    model_path = os.environ.get("GNN_MODEL_PATH", "Phoenix_project/models/gnn_model")
    try:
        logger.info(f"Worker Init: Pre-loading GNN Model Singleton from {model_path}...")
        GNNInferencer(model_path)
        logger.info("Worker Init: GNN Model successfully pre-loaded.")
    except Exception as e:
        logger.error(f"Worker Init: Failed to pre-load GNN Model: {e}", exc_info=True)


def build_orchestrator() -> Orchestrator:
    """
    [阶段 3 & 4 重构]
    初始化 Orchestrator 及其所有依赖项。
    [Fix] 这是一个工厂函数，不再使用全局单例。
    必须在 Active Event Loop 内调用以确保 Asyncio 组件正确绑定。
    """
    setup_logging()
    logger = get_logger(__name__)
    logger.info("Worker: Building new Orchestrator instance (Fresh Loop)...")
    
    try:
        # 1. Config
        config_path = os.environ.get('PHOENIX_CONFIG_PATH', 'config')
        config_loader = ConfigLoader(config_path)
        system_config = config_loader.load_config('system.yaml') 
        
        # [Task 8] Use Factory for Infrastructure
        redis_client = PhoenixFactory.create_redis_client()
        # [Critical] ContextBus initializes asyncio.Lock(), strictly requires active loop
        context_bus = PhoenixFactory.create_context_bus(redis_client, system_config.get("context_bus", {}))
        
        prompt_manager = PromptManager(prompts_dir="prompts")
        prompt_renderer = PromptRenderer(prompt_manager=prompt_manager)
        
        # 2. Data
        catalog_path = system_config.get("data_catalog_path", "data_catalog.json")
        try:
            catalog_file_path = os.path.join(config_loader.config_path, catalog_path)
            with open(catalog_file_path, 'r') as f:
                data_catalog = json.load(f)
        except FileNotFoundError:
            logger.error(f"Data catalog '{catalog_path}' not found!")
            data_catalog = {} 
            
        # 3. Core Components
        error_handler = ErrorHandler(config=system_config.get("error_handler", {}))
        
        # DB Clients
        tabular_client = TabularDBClient(
            db_uri=system_config.get("tabular_db", {}).get("uri", "sqlite:///phoenix.db"),
            llm_client=None, 
            config=system_config.get("tabular_db", {}),
            prompt_manager=prompt_manager,
            prompt_renderer=prompt_renderer
        )
        temporal_client = TemporalDBClient(config=system_config.get("temporal_db", {}))
        
        data_manager = DataManager(
            config_loader=config_loader, # Pass config_loader to match DataManager.__init__
            redis_client=redis_client
        )
        
        max_history = system_config.get("system", {}).get("max_pipeline_history", 100)
        # PipelineState is initialized in run_main_cycle or loaded
        
        audit_logger = AuditLogger(config=system_config.get('audit_db', {}))
        cot_db = CoTDatabase(config=system_config.get("ai", {}).get("cot_database", {}))
        audit_manager = AuditManager(cot_db=cot_db) # Updated to match current AuditManager signature
        snapshot_manager = SnapshotManager()

        # 4. Event Stream
        event_distributor = EventDistributor(redis_client=redis_client)
        event_filter = EventRiskFilter(config=system_config.get("events", {}).get("risk_filter", {}))

        # 5. Execution
        env = system_config.get("system", {}).get("environment", "development")
        broker_config = system_config.get("broker", {})
        broker: IBrokerAdapter

        if env == "production":
            if broker_config.get("type") == "alpaca":
                api_key = os.environ.get(broker_config.get("api_key_env"))
                api_secret = os.environ.get(broker_config.get("api_secret_env"))
                if not api_key or not api_secret:
                    raise ValueError("Missing Alpaca API credentials in environment")
                broker = AlpacaAdapter(config=broker_config) 
                # Connect happens in Orchestrator or LoopManager
            else:
                broker = SimulatedBrokerAdapter() 
        else:
            broker = SimulatedBrokerAdapter()
            
        initial_cash = system_config.get("trading", {}).get("initial_cash", 1000000)
        trade_lifecycle_manager = TradeLifecycleManager(initial_cash=initial_cash, tabular_db=tabular_client)
        order_manager = OrderManager(
            broker=broker, 
            trade_lifecycle_manager=trade_lifecycle_manager,
            data_manager=data_manager
        )

        # 6. AI/Cognitive
        gemini_pool = None 
        
        embedding_client = EmbeddingClient(
            provider="google", 
            model_name=system_config.get("ai",{}).get("embedding_client",{}).get("model", "text-embedding-004"),
            api_key=os.environ.get("GEMINI_API_KEY"),
            logger=logger
        )
        
        vector_store = VectorStore(
            config=system_config.get("ai", {}).get("vector_store", {}),
            embedding_client=embedding_client
        )
        
        knowledge_graph_service = GraphDBClient() 
        gnn_inferencer = GNNInferencer(system_config.get("ai", {}).get("gnn_inferencer", {}).get("model_path", "Phoenix_project/models/gnn_model"))

        retriever = Retriever(
            vector_store=vector_store,
            graph_db=knowledge_graph_service,
            config=system_config.get("ai", {}).get("retriever", {}),
            prompt_manager=prompt_manager,
            prompt_renderer=prompt_renderer,
            ensemble_client=None, 
            gnn_inferencer=gnn_inferencer,
            temporal_db=temporal_client,
            tabular_db=tabular_client
        )
        
        # [Task 3.3] Instantiate L2/Eval
        metacognitive_agent = MetacognitiveAgent(agent_id="meta", llm_client=None, prompt_manager=prompt_manager, prompt_renderer=prompt_renderer)
        arbitrator = Arbitrator(llm_client=None)
        fact_checker = FactChecker(llm_client=None, prompt_manager=prompt_manager, prompt_renderer=prompt_renderer)
        voter = Voter(llm_client=None)
        
        from Phoenix_project.fusion.uncertainty_guard import UncertaintyGuard
        uncertainty_guard = UncertaintyGuard()

        reasoning_ensemble = ReasoningEnsemble(
            prompt_manager=prompt_manager,
            gemini_pool=gemini_pool,
            voter=voter,
            retriever=retriever,
            ensemble_client=None,
            metacognitive_agent=metacognitive_agent,
            arbitrator=arbitrator,
            fact_checker=fact_checker,
            data_manager=data_manager
        )
        
        cognitive_engine = CognitiveEngine(
            reasoning_ensemble=reasoning_ensemble,
            fact_checker=fact_checker,
            uncertainty_guard=uncertainty_guard,
            voter=voter,
            config=system_config.get("cognitive_engine", {}),
            agent_executor=None, # Assuming Executor is not strictly needed for basic worker flow or injected later
            # Correct arguments for updated CognitiveEngine
        )
            
        # 7. Portfolio Sizer
        sizer = FixedFractionSizer(fraction_per_position=0.05)

        # 8. Risk & Portfolio
        risk_manager = RiskManager(
            config=system_config.get("trading", {}), 
            redis_client=redis_client, 
            data_manager=data_manager,
            initial_capital=initial_cash 
        )

        portfolio_constructor = PortfolioConstructor(
            config=system_config, # Or OmegaConf dict
            context_bus=context_bus,
            risk_manager=risk_manager,
            sizing_strategy=sizer,
            data_manager=data_manager
        )

        metrics_collector = MetricsCollector(config_loader)
        market_state_predictor = None 

        # 9. Create Orchestrator
        orchestrator = Orchestrator(
            config=system_config, 
            context_bus=context_bus,
            data_manager=data_manager,
            cognitive_engine=cognitive_engine,
            event_distributor=event_distributor,
            event_filter=event_filter,
            market_state_predictor=market_state_predictor,
            portfolio_constructor=portfolio_constructor,
            order_manager=order_manager,
            audit_manager=audit_manager,
            trade_lifecycle_manager=trade_lifecycle_manager,
            risk_manager=risk_manager 
        )
        
        logger.info("Worker: New Orchestrator instance built.")
        return orchestrator
    
    except Exception as e:
        logger.error(f"Worker: Failed to build Orchestrator: {e}", exc_info=True)
        raise

# --- Celery Tasks ---

async def _async_run_main_cycle():
    """
    [Phase 0 Fix] 内部 Async Entrypoint。
    在 asyncio.run() 创建的 Loop 内构建 Orchestrator，确保所有 async Primitive (Lock, Future) 
    绑定到当前 Loop，避免 'Future attached to a different loop' 错误。
    """
    orchestrator = None
    try:
        # Build inside the loop!
        orchestrator = build_orchestrator()
        await orchestrator.run_main_cycle()
    finally:
        # Cleanup: 关闭 Redis 连接防止泄漏
        if orchestrator and orchestrator.context_bus:
            # 假设 context_bus 暴露 redis client 或者有 close 方法
            if hasattr(orchestrator.context_bus, 'redis') and orchestrator.context_bus.redis:
                 await orchestrator.context_bus.redis.close()


@celery_app.task(name='phoenix.run_main_cycle')
def run_main_cycle_task():
    """
    Celery 任务，用于执行一个 Orchestrator 周期。
    """
    logger = get_logger('phoenix.run_main_cycle')
    try:
        logger.info("Task: run_main_cycle_task started...")
        
        # [阶段 4 修复] 
        # 使用 asyncio.run 启动一个新的事件循环，并调用内部 wrapper
        # 确保 Orchestrator 在此 Loop 中构建和运行
        asyncio.run(_async_run_main_cycle())
        
        logger.info("Task: run_main_cycle_task finished.")
        
    except Exception as e:
        logger.error(f"Task: run_main_cycle_task failed: {e}", exc_info=True)

# [主人喵的清洁计划 3] 新增 Janitor 任务
@celery_app.task(name='phoenix.run_system_janitor')
def run_system_janitor_task():
    logger = get_logger('phoenix.run_system_janitor')
    try:
        if run_all_cleanup_tasks:
            logger.info("Task: run_system_janitor_task started...")
            run_all_cleanup_tasks()
            logger.info("Task: run_system_janitor_task finished.")
        else:
            logger.error("Task: run_system_janitor_task failed: 'run_all_cleanup_tasks' 未能导入。")
    except Exception as e:
        logger.error(f"Task: run_system_janitor_task failed: {e}", exc_info=True)

# [Task 5B] Isolate GNN intensive tasks
@celery_app.task(name='phoenix.run_gnn_inference', queue='gnn_queue')
def run_gnn_inference_task(graph_data: dict):
    logger = get_logger('phoenix.run_gnn_inference')
    try:
        model_path = os.environ.get("GNN_MODEL_PATH", "Phoenix_project/models/gnn_model")
        inferencer = GNNInferencer(model_path)
        return asyncio.run(inferencer.infer(graph_data))
    except Exception as e:
        logger.error(f"Task: run_gnn_inference failed: {e}", exc_info=True)
        return {}

# [Task III.1] 新增 GNN 训练任务
@celery_app.task(name='phoenix.run_gnn_training')
def run_gnn_training_task():
    logger = get_logger('phoenix.run_gnn_training')
    try:
        if run_gnn_training_pipeline:
            logger.info("Task: run_gnn_training_task started...")
            run_gnn_training_pipeline()
            logger.info("Task: run_gnn_training_task finished.")
        else:
            logger.error("Task: run_gnn_training_task failed: 'run_gnn_training_pipeline' 未能导入。")
    except Exception as e:
        logger.error(f"Task: run_gnn_training_task failed: {e}", exc_info=True)

@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    # 主循环调度
    sender.add_periodic_task(
        300.0, 
        run_main_cycle_task.s(),
        name='run main cycle every 5 minutes'
    )
    # Janitor
    if run_all_cleanup_tasks:
        sender.add_periodic_task(
            crontab(hour=3, minute=0),
            run_system_janitor_task.s(),
            name='run system janitor daily'
        )
    # GNN Training
    if run_gnn_training_pipeline:
        sender.add_periodic_task(
            crontab(hour=22, minute=5),
            run_gnn_training_task.s(),
            name='run gnn training nightly'
        )

if __name__ == '__main__':
    celery_app.worker_main(argv=['worker', '--loglevel=info'])

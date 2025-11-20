"""
Celery Worker
负责异步执行 Orchestrator 任务。
"""
import os
from celery import Celery
# [主人喵的清洁计划 3] 导入 Celery 调度器
from celery.schedules import crontab

# --- 蓝图 1：导入 Celery 信号和 Prometheus 客户端 ---
from celery.signals import worker_process_init
from prometheus_client import start_http_server
# --- 结束：蓝图 1 ---

# (导入所有组件以进行初始化)
from Phoenix_project.config.loader import ConfigLoader
from Phoenix_project.data_manager import DataManager
from Phoenix_project.core.pipeline_state import PipelineState
# [主人喵的修复 2] 导入 EventDistributor
from Phoenix_project.events.event_distributor import EventDistributor
from Phoenix_project.events.risk_filter import EventRiskFilter
# [主人喵的修复 2] 移除 StreamProcessor
from Phoenix_project.cognitive.engine import CognitiveEngine
from Phoenix_project.cognitive.portfolio_constructor import PortfolioConstructor
from Phoenix_project.cognitive.risk_manager import RiskManager
from Phoenix_project.execution.order_manager import OrderManager
from Phoenix_project.execution.trade_lifecycle_manager import TradeLifecycleManager

# [阶段 3] 导入所有适配器
from Phoenix_project.execution.interfaces import IBrokerAdapter # [修复] 导入接口
from Phoenix_project.execution.adapters import SimulatedBrokerAdapter, PaperTradingBrokerAdapter
from Phoenix_project.controller.orchestrator import Orchestrator
from Phoenix_project.controller.error_handler import ErrorHandler
from Phoenix_project.audit_manager import AuditManager
from Phoenix_project.snapshot_manager import SnapshotManager
from Phoenix_project.metrics_collector import MetricsCollector
from Phoenix_project.monitor.logging import setup_logging, get_logger

# (AI/RAG components)
from Phoenix_project.ai.retriever import Retriever
from Phoenix_project.ai.ensemble_client import EnsembleClient
# [主人喵的修复 1] 修复 MetacognitiveAgent 错误的导入路径
from Phoenix_project.agents.l2.metacognitive_agent import MetacognitiveAgent 
from Phoenix_project.ai.reasoning_ensemble import ReasoningEnsemble
from Phoenix_project.evaluation.arbitrator import Arbitrator
from Phoenix_project.evaluation.fact_checker import FactChecker
from Phoenix_project.ai.prompt_manager import PromptManager
from Phoenix_project.ai.prompt_renderer import PromptRenderer # [Task 4] Import PromptRenderer
from Phoenix_project.api.gateway import APIGateway
from Phoenix_project.api.gemini_pool_manager import GeminiPoolManager
# [蓝图 2] 导入 BaseVectorStore
from Phoenix_project.memory.vector_store import get_vector_store
from Phoenix_project.memory.cot_database import CoTDatabase
# [主人喵的清洁计划 1.2] 导入新的 AuditLogger
from Phoenix_project.audit.logger import AuditLogger

# --- [主人喵的修复 1] 导入 GNN/KG 架构缺失的组件 ---
from Phoenix_project.ai.embedding_client import EmbeddingClient
from Phoenix_project.ai.tabular_db_client import TabularDBClient
from Phoenix_project.ai.temporal_db_client import TemporalDBClient
from Phoenix_project.ai.relation_extractor import RelationExtractor
from Phoenix_project.knowledge_graph_service import KnowledgeGraphService
from Phoenix_project.ai.gnn_inferencer import GNNInferencer # [Task 5A] Import for pre-loading
# --- [修复结束] ---
from Phoenix_project.context_bus import ContextBus # [Task 8] Import ContextBus

import json

# --- [主人喵的修复 2] ---
# 导入动态加载所需的模块
import importlib
import pydash # 用于将 "VolatilityParitySizer" 转换为 "volatility_parity"
import redis # [Task 8] Import redis
from Phoenix_project.sizing.base import IPositionSizer
from Phoenix_project.sizing.fixed_fraction import FixedFractionSizer # 作为安全回退
# --- [修复结束] ---

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

# [蓝图 2 修复] 导入 registry (假设它在 registry.py 中)
try:
    from Phoenix_project.registry import registry
except ImportError:
    # 提前设置日志记录器以捕获此警告
    setup_logging() 
    logger = get_logger(__name__)
    logger.warning("Could not import 'registry' from 'Phoenix_project.registry'. Using empty dict.")
    registry = {} # 回退


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

# --- 全局单例 (Global Singleton) ---
# 避免在每个 Celery 任务中都重新初始化整个应用程序
orchestrator_instance: Orchestrator = None

# --- 蓝图 1：为 Worker 启动 Prometheus 服务器 ---
@worker_process_init.connect
def start_prometheus_server(**kwargs):
    """
    在 Celery worker 进程启动时调用。
    为这个 worker 启动一个专用的 metrics http 服务器。
    """
    port = int(os.environ.get('WORKER_METRICS_PORT', 8001))
    try:
        start_http_server(port)
        print(f"Prometheus metrics server for worker started on port {port}")
    except Exception as e:
        # 可能是端口已在使用中（如果 worker 重启）
        print(f"Failed to start Prometheus metrics server on port {port}: {e}")

@worker_process_init.connect
def preload_gnn_model(**kwargs):
    """
    [Task 5A] Pre-load the GNN model as a Singleton when the worker process starts.
    This prevents resource contention and latency during the first request.
    """
    logger = get_logger(__name__)
    model_path = os.environ.get("GNN_MODEL_PATH", "Phoenix_project/models/gnn_model")
    
    try:
        logger.info(f"Worker Init: Pre-loading GNN Model Singleton from {model_path}...")
        # Instantiating the Singleton loads the model into memory once.
        # Future calls to GNNInferencer(model_path) will return this same instance.
        GNNInferencer(model_path)
        logger.info("Worker Init: GNN Model successfully pre-loaded.")
    except Exception as e:
        logger.error(f"Worker Init: Failed to pre-load GNN Model: {e}", exc_info=True)
        # We do not raise here, to allow the worker to start even if GNN fails (resilience)
# --- 结束：蓝图 1 ---


def build_orchestrator() -> Orchestrator:
    """
    [阶段 3 & 4 重构]
    初始化 Orchestrator 及其所有依赖项。
    - 实现基于 system.environment 的 Broker 依赖注入。
    - 将 TradeLifecycleManager 注入 OrderManager。
    """
    
    global orchestrator_instance
    if orchestrator_instance:
        return orchestrator_instance

    setup_logging()
    logger = get_logger(__name__)
    logger.info("Worker: Building new Orchestrator instance...")
    
    try:
        # 1. Config
        config_path = os.environ.get('PHOENIX_CONFIG_PATH', 'config')
        config_loader = ConfigLoader(config_path)
        # [修复] system.yaml 路径应相对于 config_path
        system_config = config_loader.load_config('system.yaml') 
        
        # 2. Data
        catalog_path = system_config.get("data_catalog_path", "data_catalog.json")
        try:
            # [修复] catalog_path 可能是相对路径，使用 config_loader 解析
            catalog_file_path = os.path.join(config_loader.config_path, catalog_path)
            with open(catalog_file_path, 'r') as f:
                data_catalog = json.load(f)
        except FileNotFoundError:
            logger.error(f"Data catalog '{catalog_path}' not found at '{catalog_file_path}'!")
            data_catalog = {} # 回退
            
        # 3. Core Components
        error_handler = ErrorHandler(config=system_config.get("error_handler", {}))
        data_manager = DataManager(config_loader, data_catalog)
        max_history = system_config.get("system", {}).get("max_pipeline_history", 100) # [主人喵的清洁计划 4.1] 修复路径
        pipeline_state = PipelineState(initial_state=None, max_history=max_history)
        
        # [主人喵的清洁计划 1.2/1.3] 修复 AuditManager 的初始化
        # 它需要 AuditLogger 和 CoTDatabase 实例
        audit_logger = AuditLogger(
            # [主人喵的清洁计划 4.1] 从 system.yaml 获取 'audit_db' 配置
            config=system_config.get('audit_db', {})
        )

        cot_db = CoTDatabase(
            config=system_config.get("ai", {}).get("cot_database", {}), 
            logger=logger
        )
        audit_manager = AuditManager(audit_logger=audit_logger, cot_database=cot_db)
        
        # [主人喵的清洁计划] 修复 SnapshotManager 的初始化 (它不需要参数)
        snapshot_manager = SnapshotManager()

        # 4. Event Stream
        event_distributor = EventDistributor()
        event_filter = EventRiskFilter(config_loader)

        # 5. Execution
        
        # --- [阶段 3：实现依赖注入] ---
        env = system_config.get("system", {}).get("environment", "development")
        broker_config = system_config.get("broker", {})
        broker: IBrokerAdapter

        if env == "production":
            logger.info(f"Production environment detected. Attempting to load PaperTradingBrokerAdapter.")
            if broker_config.get("type") == "alpaca":
                api_key_env = broker_config.get("api_key_env")
                api_secret_env = broker_config.get("api_secret_env")
                
                if not api_key_env or not api_secret_env:
                     logger.critical("Broker config 'api_key_env' or 'api_secret_env' missing in system.yaml")
                     raise ValueError("Missing broker API key env var names")

                api_key = os.environ.get(api_key_env)
                api_secret = os.environ.get(api_secret_env)
                
                if not api_key or not api_secret:
                    logger.critical(f"Environment variables {api_key_env} or {api_secret_env} not set.")
                    raise ValueError("Missing Alpaca API credentials in environment")
                    
                broker = PaperTradingBrokerAdapter(
                    api_key=api_key,
                    api_secret=api_secret,
                    paper_base_url=broker_config.get("base_url") # 确保这是 paper URL
                )
                broker.connect() # 在启动时连接
            else:
                logger.critical("Production environment selected, but no valid 'alpaca' broker configured.")
                raise ValueError("Production environment requires a valid broker configuration.")
        else:
            logger.info(f"Development environment detected. Loading SimulatedBrokerAdapter.")
            broker = SimulatedBrokerAdapter()
            # (SimBroker 不再需要 data_manager)
            
        # --- [阶段 4：注入 TLM 到 OrderManager] ---
        initial_cash = system_config.get("trading", {}).get("initial_cash", 1000000)
        trade_lifecycle_manager = TradeLifecycleManager(initial_cash=initial_cash)
        order_manager = OrderManager(
            broker=broker, 
            trade_lifecycle_manager=trade_lifecycle_manager # <--- 注入
        )
        # --- [结束 阶段 3 & 4] ---

        # 6. AI/Cognitive (与 phoenix_project.py 相同)
        gemini_pool = GeminiPoolManager()
        api_gateway = APIGateway(gemini_pool)
        prompt_manager = PromptManager(config_loader.config_path)
        prompt_renderer = PromptRenderer(prompt_manager) # [Task 4] Instantiate PromptRenderer
        
        embedding_client = EmbeddingClient(
            model_name=system_config.get("ai",{}).get("embedding_client",{}).get("model", "text-embedding-004"),
            logger=logger # [主人喵的清洁计划 1.1 修复] 传递 logger
        )
        tabular_client = TabularDBClient(config=system_config.get("tabular_db", {}))
        temporal_client = TemporalDBClient(config=system_config.get("temporal_db", {}))
        relation_extractor = RelationExtractor(api_gateway, prompt_manager)
        
        vector_store = get_vector_store(
            config=system_config.get("ai", {}).get("vector_database", {}),
            embedding_client=embedding_client,
            logger=logger
        )
        
        # (cot_db 已在上面初始化)
        
        knowledge_graph_service = KnowledgeGraphService(config=system_config.get("neo4j_db", {}))
        
        # [Task P-5.1] Instantiate GNNInferencer (Singleton)
        model_path = system_config.get("ai", {}).get("gnn_inferencer", {}).get("model_path", "Phoenix_project/models/gnn_model")
        gnn_inferencer = GNNInferencer(model_path)

        # [Task P-5.2] Fix Retriever Instantiation
        retriever_config = system_config.get("ai", {}).get("retriever", {})
        
        # [Task 1.3] Move EnsembleClient init BEFORE Retriever to fix dependency injection
        agent_registry_config = config_loader.get_agent_registry()
        ensemble_client = EnsembleClient(gemini_pool, prompt_manager, agent_registry_config, registry)
        
        retriever = Retriever(
            vector_store=vector_store,
            graph_db=knowledge_graph_service,
            config=retriever_config,
            prompt_manager=prompt_manager,
            prompt_renderer=prompt_renderer, # [Task 4] Correctly pass PromptRenderer instance
            ensemble_client=ensemble_client, # [Task 1.3] Inject initialized client
            gnn_inferencer=gnn_inferencer,
            temporal_db=temporal_client,
            tabular_db=tabular_client
        )
        
        # [Task 5] Fix MetacognitiveAgent params (kwargs + prompt_renderer)
        metacognitive_agent = MetacognitiveAgent(
            agent_id="metacognitive_agent",
            llm_client=api_gateway,
            prompt_manager=prompt_manager,
            prompt_renderer=prompt_renderer
        )
        
        # [Task 6] Fix Arbitrator/FactChecker params (add prompt_renderer + kwargs)
        arbitrator = Arbitrator(
            llm_client=api_gateway,
            prompt_manager=prompt_manager,
            prompt_renderer=prompt_renderer
        )
        fact_checker = FactChecker(
            llm_client=api_gateway,
            prompt_manager=prompt_manager,
            prompt_renderer=prompt_renderer
        )
        
        # [Task 7] Fix ReasoningEnsemble params (add missing deps)
        # [修复] 导入 CognitiveEngine 缺少的依赖
        from Phoenix_project.fusion.uncertainty_guard import UncertaintyGuard
        from Phoenix_project.evaluation.voter import Voter
        uncertainty_guard = UncertaintyGuard()
        voter = Voter()

        reasoning_ensemble = ReasoningEnsemble(
            prompt_manager=prompt_manager,
            gemini_pool=gemini_pool,
            voter=voter,
            retriever=retriever,
            ensemble_client=ensemble_client,
            metacognitive_agent=metacognitive_agent,
            arbitrator=arbitrator,
            fact_checker=fact_checker
        )
        
        cognitive_engine = CognitiveEngine(
            reasoning_ensemble=reasoning_ensemble,
            fact_checker=fact_checker,
            uncertainty_guard=uncertainty_guard, # [修复]
            voter=voter, # [修复]
            config=system_config.get("cognitive_engine", {}) # [修复]
        )
            
        # 7. Portfolio
        sizer_config = system_config.get("portfolio", {}).get("sizer", {})
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
            logger.info(f"成功从配置加载 Sizer: {sizer_type_name} (Params: {sizer_params})")
        except (ImportError, AttributeError, TypeError) as e:
            logger.error(f"无法从配置加载 sizer '{sizer_type_name}' (Params: {sizer_params}): {e}。回退到 FixedFractionSizer。")
            sizer = FixedFractionSizer(fraction_per_position=0.05)
        
        # [Task 8] Fix RiskManager and PortfolioConstructor dependency order and params
        
        # 8.1 Initialize Infrastructure
        redis_host = os.environ.get("REDIS_HOST", "localhost")
        redis_port = int(os.environ.get("REDIS_PORT", 6379))
        redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)
        context_bus = ContextBus()

        # 8.2 Initialize RiskManager FIRST (Dependency)
        risk_manager = RiskManager(
            config=system_config, 
            redis_client=redis_client, 
            initial_capital=initial_cash # Reusing initial_cash from Section 5
        )

        # 8.3 Initialize PortfolioConstructor SECOND (Dependent)
        portfolio_constructor = PortfolioConstructor(
            config=system_config,
            context_bus=context_bus,
            risk_manager=risk_manager,
            sizing_strategy=sizer,
            data_manager=data_manager
        )

        # 8. Monitoring
        metrics_collector = MetricsCollector(config_loader)

        # 9. 创建 Orchestrator 实例
        orchestrator_instance = Orchestrator(
            pipeline_state=pipeline_state,
            data_manager=data_manager,
            event_filter=event_filter,
            event_distributor=event_distributor,
            cognitive_engine=cognitive_engine,
            portfolio_constructor=portfolio_constructor,
            risk_manager=risk_manager,
            order_manager=order_manager,
            trade_lifecycle_manager=trade_lifecycle_manager,
            snapshot_manager=snapshot_manager,
            metrics_collector=metrics_collector,
            audit_manager=audit_manager,
            error_handler=error_handler
        )
        
        logger.info("Worker: New Orchestrator instance built and cached.")
        return orchestrator_instance
    
    except Exception as e:
        logger.error(f"Worker: Failed to build Orchestrator: {e}", exc_info=True)
        raise

# --- Celery Tasks ---

@celery_app.task(name='phoenix.run_main_cycle')
def run_main_cycle_task():
    """
    Celery 任务，用于执行一个 Orchestrator 周期。
    """
    logger = get_logger('phoenix.run_main_cycle')
    try:
        logger.info("Task: run_main_cycle_task started...")
        orchestrator = build_orchestrator()
        
        # [阶段 4] run_main_cycle 是同步的
        orchestrator.run_main_cycle() 
        
        logger.info("Task: run_main_cycle_task finished.")
        
    except Exception as e:
        logger.error(f"Task: run_main_cycle_task failed: {e}", exc_info=True)
        # (可选: 重试)
        # raise self.retry(exc=e, countdown=60)

# [主人喵的清洁计划 3] 新增 Janitor 任务
@celery_app.task(name='phoenix.run_system_janitor')
def run_system_janitor_task():
    """[主人喵的清洁计划 3] Celery 任务，用于执行系统清理。"""
    logger = get_logger('phoenix.run_system_janitor')
    try:
        if run_all_cleanup_tasks:
            logger.info("Task: run_system_janitor_task started...")
            run_all_cleanup_tasks() # <--- [主人喵的清洁计划 3] 直接调用
            logger.info("Task: run_system_janitor_task finished.")
        else:
            logger.error("Task: run_system_janitor_task failed: 'run_all_cleanup_tasks' 未能导入。")
            
    except Exception as e:
        logger.error(f"Task: run_system_janitor_task failed: {e}", exc_info=True)


# [Task 5B] Isolate GNN intensive tasks
@celery_app.task(name='phoenix.run_gnn_inference', queue='gnn_queue')
def run_gnn_inference_task(graph_data: dict):
    """
    [Task 5B] Dedicated task for running GNN inference.
    Runs on the 'gnn_queue' to avoid blocking main agents.
    """
    logger = get_logger('phoenix.run_gnn_inference')
    try:
        # Use the Singleton instance (pre-loaded by worker_process_init)
        model_path = os.environ.get("GNN_MODEL_PATH", "Phoenix_project/models/gnn_model")
        inferencer = GNNInferencer(model_path)
        
        # Run inference synchronously within this worker process
        # (Note: .infer() is async, so we run it in a loop if needed, 
        # but since this is a Celery task, we can just run it. 
        # However, GNNInferencer.infer is async def. We need a loop.)
        import asyncio
        return asyncio.run(inferencer.infer(graph_data))
        
    except Exception as e:
        logger.error(f"Task: run_gnn_inference failed: {e}", exc_info=True)
        return {}


# [Task III.1] 新增 GNN 训练任务
@celery_app.task(name='phoenix.run_gnn_training')
def run_gnn_training_task():
    """[Task III.1] Celery 任务，用于执行夜间 GNN 训练。"""
    logger = get_logger('phoenix.run_gnn_training')
    try:
        if run_gnn_training_pipeline:
            logger.info("Task: run_gnn_training_task started...")
            run_gnn_training_pipeline() # <--- [Task III.1] 调用 GNN 引擎
            logger.info("Task: run_gnn_training_task finished.")
        else:
            logger.error("Task: run_gnn_training_task failed: 'run_gnn_training_pipeline' 未能导入。")
            
    except Exception as e:
        logger.error(f"Task: run_gnn_training_task failed: {e}", exc_info=True)


@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    """
    [主人喵的清洁计划 4.1] [已更新]
    使用 Celery Beat 设置定时任务。
    这替代了 controller/scheduler.py。
    """
    
    # [主人喵的最终优化] 添加主循环调度 (每 5 分钟，来自旧的 system.yaml)
    sender.add_periodic_task(
        300.0, # 300 秒 = 5 分钟
        run_main_cycle_task.s(),
        name='run main cycle every 5 minutes'
    )
    
    # [主人喵的清洁计划 4.1] 每天凌晨 3:00 运行 Janitor
    if run_all_cleanup_tasks:
        sender.add_periodic_task(
            crontab(hour=3, minute=0), # 每天 3:00 AM
            run_system_janitor_task.s(),
            name='run system janitor daily'
        )

    # [Task III.1] 每天 22:05 UTC 运行 GNN 训练
    if run_gnn_training_pipeline:
        sender.add_periodic_task(
            crontab(hour=22, minute=5), # 每天 22:0D5 UTC
            run_gnn_training_task.s(),
            name='run gnn training nightly'
        )


if __name__ == '__main__':
    # 直接运行 worker (用于开发)
    celery_app.worker_main(argv=['worker', '--loglevel=info'])

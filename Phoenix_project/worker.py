"""
Celery Worker
负责异步执行 Orchestrator 任务。
"""
import os
from celery import Celery

# (导入所有组件以进行初始化)
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
from Phoenix_project.controller.error_handler import ErrorHandler
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
from Phoenix_project.ai.prompt_manager import PromptManager
from Phoenix_project.api.gateway import APIGateway
from Phoenix_project.api.gemini_pool_manager import GeminiPoolManager
from Phoenix_project.memory.vector_store import VectorStore
from Phoenix_project.memory.cot_database import CoTDatabase
from Phoenix_project.sizing.fixed_fraction import FixedFractionSizer
import json

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

def build_orchestrator() -> Orchestrator:
    """
    FIX (E5): 重写此函数以正确初始化 Orchestrator 及其所有依赖项。
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
        
        # 2. Data
        catalog_path = config_loader.get_system_config().get("data_catalog_path", "data_catalog.json")
        with open(catalog_path, 'r') as f:
            data_catalog = json.load(f)
            
        # 3. Core Components
        error_handler = ErrorHandler()
        
        # (E5 Fix)
        data_manager = DataManager(config_loader, data_catalog)
        
        # (E5 Fix)
        max_history = config_loader.get_system_config().get("max_pipeline_history", 100)
        pipeline_state = PipelineState(initial_state=None, max_history=max_history)
        
        audit_manager = AuditManager(config_loader)
        snapshot_manager = SnapshotManager(config_loader)

        # 4. Event Stream
        event_filter = EventRiskFilter(config_loader)
        stream_processor = StreamProcessor()

        # 5. Execution
        # (E6 Fix - 检查 Alpaca)
        broker_config = config_loader.get_system_config().get("broker", {})
        if broker_config.get("type") == "alpaca":
            broker = AlpacaAdapter(
                api_key=os.environ.get(broker_config.get("api_key_env")),
                api_secret=os.environ.get(broker_config.get("api_secret_env")),
                base_url=broker_config.get("base_url")
            )
        else:
            logger.warning("Defaulting to SimulatedBrokerAdapter for worker.")
            broker = SimulatedBrokerAdapter()
            
        initial_cash = config_loader.get_system_config().get("trading", {}).get("initial_cash", 1000000)
        trade_lifecycle_manager = TradeLifecycleManager(initial_cash=initial_cash)
        order_manager = OrderManager(broker)

        # 6. AI/Cognitive (与 phoenix_project.py 相同)
        gemini_pool = GeminiPoolManager()
        api_gateway = APIGateway(gemini_pool)
        prompt_manager = PromptManager(config_loader.config_path)
        vector_store = VectorStore()
        cot_db = CoTDatabase()
        retriever = Retriever(vector_store, cot_db)
        agent_registry = config_loader.get_agent_registry()
        ensemble_client = EnsembleClient(api_gateway, prompt_manager, agent_registry)
        metacognitive_agent = MetacognitiveAgent(api_gateway, prompt_manager)
        arbitrator = Arbitrator(api_gateway, prompt_manager)
        fact_checker = FactChecker(api_gateway, prompt_manager)
        
        reasoning_ensemble = ReasoningEnsemble(
            retriever=retriever,
            ensemble_client=ensemble_client,
            metacognitive_agent=metacognitive_agent,
            arbitrator=arbitrator,
            fact_checker=fact_checker
        )
        
        cognitive_engine = CognitiveEngine(
            reasoning_ensemble=reasoning_ensemble,
            fact_checker=fact_checker
        )
            
        # 7. Portfolio
        sizer = FixedFractionSizer() # 示例
        portfolio_constructor = PortfolioConstructor(position_sizer=sizer)
        risk_manager = RiskManager(config_loader)

        # 8. Monitoring
        metrics_collector = MetricsCollector(config_loader)

        # 9. 创建 Orchestrator 实例
        orchestrator_instance = Orchestrator(
            pipeline_state=pipeline_state,
            data_manager=data_manager,
            event_filter=event_filter,
            stream_processor=stream_processor,
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
        # (在 Celery 中，我们可能希望任务失败而不是让 worker 崩溃)
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
        # (获取或构建单例)
        orchestrator = build_orchestrator()
        
        # FIX (E7): 调用 run_main_cycle() 而不是 run_main_loop_async()
        orchestrator.run_main_cycle() 
        
        logger.info("Task: run_main_cycle_task finished.")
        
    except Exception as e:
        logger.error(f"Task: run_main_cycle_task failed: {e}", exc_info=True)
        # (可选: 重试)
        # raise self.retry(exc=e, countdown=60)

@celery_app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    """
    (可选) 如果使用 Celery Beat，可以在这里设置定时任务。
    这替代了 controller/scheduler.py。
    """
    # 示例：每5分钟运行一次
    # sender.add_periodic_task(
    #     300.0,
    #     run_main_cycle_task.s(),
    #     name='Run main cycle every 5 minutes'
    # )
    pass

if __name__ == '__main__':
    # 直接运行 worker (用于开发)
    # celery -A worker worker --loglevel=info
    celery_app.worker_main(argv=['worker', '--loglevel=info'])

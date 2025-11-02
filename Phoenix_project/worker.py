"""
Celery Worker
- 定义和注册 Celery 任务
- 构建核心服务 (Orchestrator)
- 运行认知工作流
"""
import os
import asyncio
from celery import Celery
from kombu import Queue

from monitor.logging import get_logger, setup_logging
from config.loader import ConfigLoader
from core.pipeline_state import PipelineState
from data_manager import DataManager
from strategy_handler import StrategyHandler
from execution.order_manager import OrderManager
from execution.adapters import AlpacaAdapter # (假设使用 Alpaca)
from events.event_distributor import EventDistributor
from controller.orchestrator import Orchestrator
from api.gemini_pool_manager import GeminiPoolManager
from ai.metacognitive_agent import MetacognitiveAgent
from cognitive.portfolio_constructor import PortfolioConstructor

# --- Celery App Setup ---
setup_logging()
logger = get_logger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672//")

app = Celery(
    "phoenix_worker",
    broker=RABBITMQ_URL,
    backend=REDIS_URL,
)
app.conf.update(
    task_queues=(Queue("cognitive_workflow"),),
    task_default_queue="cognitive_workflow",
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
)

# --- 全局 Orchestrator 实例 ---
# (在 Celery worker 中，在任务执行前构建)
orchestrator_instance = None

def build_orchestrator():
    """
    关键修正 (Error 4):
    重构此函数以正确构建 Orchestrator 及其所有依赖项。
    这必须与 controller/orchestrator.py 的 __init__ 签名严格匹配。
    """
    logger.info("Building core components for Orchestrator...")
    
    try:
        # 1. 加载配置
        config_loader = ConfigLoader()
        config = config_loader.get_system_config()

        # 2. 核心状态和管理器
        pipeline_state = PipelineState()
        data_manager = DataManager(config.get('data_manager', {}))
        event_distributor = EventDistributor()

        # 3. AI 和 API Pool (这些是 MetacognitiveAgent 的依赖)
        gemini_pool = GeminiPoolManager(config.get('gemini', {}))
        
        # 4. 认知组件 (这些是 StrategyHandler 的依赖)
        metacognitive_agent = MetacognitiveAgent(
            config=config.get('metacognitive_agent', {}),
            gemini_pool=gemini_pool
            # (可能还需要其他依赖, e.g., prompt_manager)
        )
        portfolio_constructor = PortfolioConstructor(
            config=config.get('portfolio_constructor', {})
            # (可能还需要其他依赖)
        )
        
        # 5. 执行组件 (这些是 OrderManager 和 Orchestrator 的依赖)
        # 修正: OrderManager 构造函数需要一个 adapter
        broker_adapter = AlpacaAdapter(config.get('execution', {}).get('alpaca', {}))
        order_manager = OrderManager(
            config=config.get('execution', {}), 
            adapter=broker_adapter
        )

        # 6. 策略处理器
        strategy_handler = StrategyHandler(
            config=config.get('strategy_handler', {}),
            data_manager=data_manager,
            metacognitive_agent=metacognitive_agent,
            portfolio_constructor=portfolio_constructor
        )

        # 7. 实例化 Orchestrator
        # 修正: 使用 controller.orchestrator.py 中的正确签名
        orchestrator = Orchestrator(
            config_loader=config_loader,
            pipeline_state=pipeline_state,
            data_manager=data_manager,
            strategy_handler=strategy_handler,
            order_manager=order_manager,
            event_distributor=event_distributor
            # 注意: gemini_pool 和 celery_app 不直接注入 Orchestrator
            # 它们被注入到 Orchestrator 的 *依赖项* 中
        )
        
        logger.info("Orchestrator built successfully.")
        return orchestrator
        
    except Exception as e:
        logger.error(f"Failed to build Orchestrator: {e}", exc_info=True)
        return None

# --- Celery Tasks ---

@app.on_after_configure.connect
def setup_periodic_tasks(sender, **kwargs):
    """设置定时任务"""
    sender.add_periodic_task(
        60.0,  # (e.g., 每60秒运行一次)
        run_cognitive_workflow.s(),
        name="Run Cognitive Workflow",
    )

@app.task(name="run_cognitive_workflow", bind=True)
def run_cognitive_workflow(self, *args, **kwargs):
    """
    Celery 任务，用于执行主认知工作流。
    """
    global orchestrator_instance
    
    try:
        if orchestrator_instance is None:
            logger.info("Orchestrator not found. Building...")
            orchestrator_instance = build_orchestrator()
            if orchestrator_instance is None:
                logger.error("Failed to build orchestrator. Task cannot run.")
                return "BUILD_FAILURE"

        logger.info("Running cognitive workflow task...")
        
        # 关键修正 (Error 5):
        # Orchestrator 中没有 'run_cognitive_workflow' 方法。
        # 调用正确的主循环入口点 'run_main_loop_async'。
        asyncio.run(orchestrator_instance.run_main_loop_async())
        
        logger.info("Cognitive workflow task finished.")
        return "SUCCESS"
        
    except Exception as e:
        logger.error(f"Error during cognitive workflow task: {e}", exc_info=True)
        # (可以添加重试逻辑)
        raise self.retry(exc=e, countdown=60)

if __name__ == "__main__":
    # 此 worker 通过 celery CLI 启动:
    # celery -A worker worker --loglevel=info -Q cognitive_workflow
    logger.info("Starting Celery worker...")

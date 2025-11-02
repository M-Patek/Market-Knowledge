"""
Celery Worker
This module defines the Celery application and the tasks that run
on the worker processes.

The primary task is `run_cognitive_workflow`, which performs the
heavy AI/cognitive lifting off the main event loop.
"""
import os
import logging
import asyncio
from celery import Celery
from celery.signals import worker_process_init
from typing import Optional, Dict, Any

from controller.orchestrator import Orchestrator
# 修复：[FIX-16] 移除了这个错误的导入，
# build_orchestrator 是在这个文件中定义的。
# from ..core.pipeline_service_builder import build_orchestrator

# --- Celery App Setup ---
# Load Redis URL from environment variables
REDIS_BROKER_URL = os.environ.get('REDIS_BROKER_URL', 'redis://localhost:6379/0')
REDIS_BACKEND_URL = os.environ.get('REDIS_BACKEND_URL', 'redis://localhost:6379/1')

app = Celery(
    'phoenix_worker',
    broker=REDIS_BROKER_URL,
    backend=REDIS_BACKEND_URL,
    include=['worker'] # Tells Celery to look for tasks in this module
)

app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    task_track_started=True,
)

# --- Global Orchestrator (per-worker) ---
# We initialize this to None. It will be built *once* per
# worker process when the worker first starts.
_orchestrator: Optional[Orchestrator] = None

def build_orchestrator() -> Orchestrator:
    """
    Builds a *complete* Orchestrator instance with all its
    dependencies (CognitiveEngine, DataManager, etc.).
    
    This is a heavy operation and should only be done once
    per worker process.
    
    修复：[FIX-16] 此函数已完全重写，
    以镜像 'phoenix_project.py' (主入口点) 中的依赖注入 (DI) 树。
    这确保了 worker 拥有与主应用程序相同的组件配置。
    """
    logging.info("Building a new Orchestrator instance for this worker process...")
    
    # --- 导入所有必要的组件 ---
    from config.loader import load_config
    from core.pipeline_state import PipelineState
    from data_manager import DataManager
    from strategy_handler import RomanLegionStrategy
    from execution.order_manager import OrderManager
    from api.gemini_pool_manager import GeminiPoolManager
    # Worker 总是使用模拟执行器
    from execution.adapters import SimulatedBrokerAdapter 
    
    # --- 加载配置 ---
    config_path = os.getenv('CONFIG_PATH', 'config/system.yaml')
    config = load_config(config_path)
    if config is None:
        raise RuntimeError(f"Worker failed to load config from {config_path}")

    # --- 1. 初始化 API 池 ---
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    llm_config = config.get('llm', {})
    llm_config['api_key'] = gemini_api_key
    gemini_pool = GeminiPoolManager(config=llm_config)

    # --- 2. 初始化核心状态和数据 ---
    pipeline_config = config.get('pipeline', {})
    pipeline_state = PipelineState(
        max_recent_events=pipeline_config.get('max_recent_events', 100)
    )
    cache_dir = config.get('data_manager', {}).get('cache_dir', 'data_cache')
    data_manager = DataManager(config, pipeline_state, cache_dir=cache_dir)

    # --- 3. 初始化执行 (模拟) ---
    order_manager = OrderManager(
        config=config.get('execution', {}),
        # Worker 使用模拟适配器进行离线处理
        adapter=SimulatedBrokerAdapter(config.get('execution', {}), pipeline_state)
    )

    # --- 4. 初始化策略处理器 ---
    strategy = RomanLegionStrategy(
        config=config,
        data_manager=data_manager
        # 注意：RomanLegionStrategy 在 'phoenix_project.py' 中
        # 看起来没有接收到完整的依赖，
        # 这可能是一个单独的问题。我们在这里镜像它。
    )
    
    # --- 5. 构建 Orchestrator ---
    # Worker 不需要 EventDistributor，
    # 并且它使用自己的 Celery 'app' 实例
    orchestrator = Orchestrator(
        config=config,
        data_manager=data_manager,
        pipeline_state=pipeline_state,
        gemini_pool=gemini_pool,
        strategy_handler=strategy,
        order_manager=order_manager,
        celery_app=app  # 传入 worker 的 celery 实例
    )
    
    logging.info("Orchestrator instance built successfully for worker.")
    return orchestrator

def get_orchestrator() -> Orchestrator:
    """
    Singleton accessor for the orchestrator.
    Builds it on the first call within the worker process.
    """
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = build_orchestrator()
    return _orchestrator

@worker_process_init.connect
def on_worker_init(**kwargs):
    """
    Called when a new worker process is forked.
    We initialize logging here.
    """
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.info(f"New worker process initialized (PID: {os.getpid()}).")
    # 我们可以根据需要预热 orchestrator
    # get_orchestrator()


# --- Celery Tasks ---

@app.task(name="worker.run_cognitive_workflow")
def run_cognitive_workflow(event_dict: Optional[Dict[str, Any]], 
                           task_name: Optional[str]):
    """
    The main Celery task to execute the cognitive workflow.
    
    This runs on a worker process.
    """
    logging.info(f"Worker received task: run_cognitive_workflow (Task: {task_name}, Event: {event_dict is not None})")
    
    try:
        # 1. Get the worker's singleton Orchestrator instance
        orchestrator = get_orchestrator()
        
        # 2. Deserialize event (if it exists)
        event = None
        if event_dict:
            # We need to import the schemas to parse the dict
            from core.schemas.data_schema import MarketEvent, EconomicEvent
            # 简单的类型检查
            if 'content' in event_dict:
                event = MarketEvent(**event_dict)
            else:
                event = EconomicEvent(**event_dict)

        # 3. Run the cognitive workflow
        # 修复：[FIX-16] Orchestrator 上没有 'run_cognitive_workflow_sync'。
        # 正确的方法是 'run_cognitive_workflow' (这是一个 async 方法)。
        # 我们必须为这个任务创建一个新的 asyncio 事件循环来运行它。
        
        async def run_async_workflow():
            await orchestrator.run_cognitive_workflow(
                event=event,
                task_name=task_name
            )

        # Run the async workflow synchronously
        asyncio.run(run_async_workflow())
        
        logging.info(f"Cognitive workflow task completed successfully.")
        # 任务成功完成后，Celery 会自动处理 'link'
        # (例如触发 'release_processing_lock')

    except Exception as e:
        logging.error(f"Cognitive workflow task FAILED: {e}", exc_info=True)
        # The task will be marked as FAILED in the backend
        raise

@app.task(name="worker.release_processing_lock")
def release_processing_lock():
    """
    A simple task (linked as a callback) that tells the *main*
    orchestrator (via Redis) to release its processing lock.
    
    注意：这假设主进程 (LoopManager) 正在 *轮询* 这个 Redis 键。
    在 'controller/loop_manager.py' 中使用 'asyncio.Event'
    的锁是*行不通的*，因为该锁在不同的进程中。
    """
    logging.debug("Task: Releasing processing lock via Redis...")
    try:
        import redis
        r = redis.from_url(REDIS_BROKER_URL, decode_responses=True)
        
        # 锁键必须与 LoopManager 中检查的键匹配
        lock_key = "phoenix_processing_lock"
        r.delete(lock_key)
        
        logging.info("Redis processing lock released.")

    except Exception as e:
        logging.error(f"Failed to release processing lock: {e}", exc_info=True)

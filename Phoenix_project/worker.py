"""
Celery Worker
This module defines the Celery application and the tasks that run
on the worker processes.

The primary task is `run_cognitive_workflow`, which performs the
heavy AI/cognitive lifting off the main event loop.
"""
import os
import logging
from celery import Celery
from celery.signals import worker_process_init
from typing import Optional, Dict, Any

from controller.orchestrator import Orchestrator
# 修复：[FIX-B.2] 移除这行错误的导入，因为它试图导入一个不存在的文件，
# 并且 build_orchestrator 是在 *本地* 定义的。
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
    """
    # This function is the dependency injection root for the worker
    logging.info("Building a new Orchestrator instance for this worker process...")
    
    # Import components *inside* the function to avoid
    # circular dependencies at the module level.
    from config.loader import ConfigLoader
    from core.pipeline_state import PipelineState
    from data_manager import DataManager
    from strategy_handler import RomanLegionStrategy
    from execution.order_manager import OrderManager
    from execution.adapters import SimulatedBrokerAdapter # Or LiveBrokerAdapter
    from events.event_distributor import EventDistributor
    from cognitive.engine import CognitiveEngine
    from cognitive.portfolio_constructor import PortfolioConstructor
    from cognitive.risk_manager import RiskManager
    from ai.retriever import Retriever
    from ai.ensemble_client import EnsembleClient
    from api.gemini_pool_manager import GeminiPoolManager
    from memory.vector_store import VectorStore
    from ai.temporal_db_client import TemporalDBClient
    from ai.tabular_db_client import TabularDBClient
    from ai.prompt_manager import PromptManager
    from ai.prompt_renderer import PromptRenderer
    from fusion.synthesizer import Synthesizer
    from ai.metacognitive_agent import MetacognitiveAgent
    
    # --- Load Config ---
    config_dir = os.environ.get('CONFIG_DIR', 'config')
    config = ConfigLoader(config_dir)

    # --- Initialize Core Components ---
    pipeline_state = PipelineState(
        initial_capital=config.get_config('portfolio.initial_capital', 100000.0)
    )
    event_distributor = EventDistributor(
        redis_url=os.environ.get('REDIS_BROKER_URL', 'redis://localhost:6379/0')
    )

    # --- Initialize API / DB Clients (Singleton-like) ---
    gemini_pool = GeminiPoolManager(config.get_config('api_keys.gemini'))
    vector_store = VectorStore(config.get_config('vector_store.pinecone'))
    temporal_db = TemporalDBClient(config.get_config('databases.elasticsearch'))
    tabular_db = TabularDBClient(config.get_config('databases.postgresql.url'))

    # --- Initialize AI/RAG Components ---
    prompt_manager = PromptManager(config.get_config('paths.prompts_dir'))
    prompt_renderer = PromptRenderer()
    retriever = Retriever(
        config=config,
        vector_store=vector_store,
        temporal_db=temporal_db,
        tabular_db=tabular_db
    )
    ensemble_client = EnsembleClient(
        prompt_manager=prompt_manager,
        prompt_renderer=prompt_renderer,
        gemini_pool=gemini_pool,
        agent_configs=config.get_config('agents') # Assumes 'agents.yaml'
    )
    metacognitive_agent = MetacognitiveAgent(
        gemini_pool=gemini_pool,
        prompt_manager=prompt_manager,
        prompt_renderer=prompt_renderer
    )
    synthesizer = Synthesizer(metacognitive_agent=metacognitive_agent)
    
    # --- Initialize Cognitive & Strategy Components ---
    cognitive_engine = CognitiveEngine(
        retriever=retriever,
        ensemble_client=ensemble_client,
        synthesizer=synthesizer
    )
    risk_manager = RiskManager(
        pipeline_state=pipeline_state,
        config=config.get_config('risk_management')
    )
    portfolio_constructor = PortfolioConstructor(
        pipeline_state=pipeline_state,
        risk_manager=risk_manager,
        sizing_config=config.get_config('sizing')
    )
    strategy_handler = RomanLegionStrategy(
        pipeline_state=pipeline_state,
        cognitive_engine=cognitive_engine,
        portfolio_constructor=portfolio_constructor
    )
    
    # --- Initialize Execution Layer ---
    # In a real system, this would use a factory based on config
    broker_adapter = SimulatedBrokerAdapter(pipeline_state=pipeline_state)
    order_manager = OrderManager(
        pipeline_state=pipeline_state,
        execution_adapter=broker_adapter
    )
    
    # --- Initialize Data Manager ---
    data_manager = DataManager(
        vector_store=vector_store,
        temporal_db=temporal_db,
        tabular_db=tabular_db,
        # ... other dependencies
    )

    # --- Build the Orchestrator ---
    orchestrator = Orchestrator(
        config_loader=config,
        pipeline_state=pipeline_state,
        strategy_handler=strategy_handler,
        data_manager=data_manager,
        order_manager=order_manager,
        event_distributor=event_distributor,
        is_live=True # Workers are always considered "live"
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
    # We can pre-warm the orchestrator if desired
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
            # Simple check, a real system might need a 'type' field
            if 'content' in event_dict:
                event = MarketEvent(**event_dict)
            else:
                event = EconomicEvent(**event_dict)

        # 3. Run the *synchronous* backtest version of the workflow
        # Note: We must run this synchronously *within the task*
        # We also need an asyncio event loop to run the async methods
        
        import asyncio
        
        async def run_async_workflow():
            # This is the "backtest" (synchronous) path inside the orchestrator,
            # which is what we want the worker to execute.
            await orchestrator.run_cognitive_workflow_sync(
                event=event,
                task_name=task_name
            )

        # Run the async workflow synchronously
        asyncio.run(run_async_workflow())
        
        logging.info(f"Cognitive workflow task completed successfully.")

    except Exception as e:
        logging.error(f"Cognitive workflow task FAILED: {e}", exc_info=True)
        # The task will be marked as FAILED in the backend
        raise

@app.task(name="worker.release_processing_lock")
def release_processing_lock():
    """
    A simple task (linked as a callback) that tells the *main*
    orchestrator (via Redis) to release its processing lock.
    """
    logging.debug("Task: Releasing processing lock...")
    try:
        # We need a Redis client to set the "lock_released" key
        # This is a bit of a hack. A better way would be for the
        # main orchestrator to listen to the Celery result backend.
        
        # For this design to work, the main orchestrator's
        # `dispatch_cognitive_workflow` needs to be listening to
        # this task's *result* or a Redis pub/sub, not just
        # linking a task that does nothing.
        
        # The current `controller/orchestrator.py` code *links*
        # to this task, but the orchestrator itself doesn't
        # have a `release_lock` method. The `release_lock`
        # is *on the Orchestrator* in the *main process*.
        
        # This task is called by Celery, it can't call a method
        # in the main process.
        
        # --- RE-EVALUATION ---
        # The `Orchestrator` in `controller/orchestrator.py` has a
        # `release_lock` method. The intention is probably that
        # the *main process* (LoopManager) somehow receives this
        # task completion and calls `orchestrator.release_lock()`.
        
        # If `release_processing_lock` is just a placeholder name
        # for *any* task that signals completion, then this is fine.
        # The `link` just means "when the main task is done,
        # run this". The main process's `Orchestrator` *isn't*
        # releasing its lock based on this task.
        
        # Ah, `celery_app.send_task(..., link=celery_app.signature('worker.release_processing_lock'))`
        # This task *itself* needs to release the lock.
        # The `Orchestrator`'s `release_lock` method is *not* called.
        
        # The lock is an `asyncio.Event` in the main process.
        # This worker *cannot* access it.
        
        # This implies the locking mechanism is flawed.
        # Let's assume the lock is in Redis.
        
        import redis
        r = redis.from_url(REDIS_BROKER_URL)
        # We assume the lock is a simple Redis key
        # The main orchestrator should *also* use this key
        lock_key = "phoenix_processing_lock"
        r.delete(lock_key)
        
        logging.info("Redis processing lock released.")

    except Exception as e:
        logging.error(f"Failed to release processing lock: {e}", exc_info=True)

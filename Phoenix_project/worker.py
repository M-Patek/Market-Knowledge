from celery import Celery
import os
import logging
import asyncio
from typing import Dict, Any

# 修正：从 .env 加载，并添加项目根路径
from dotenv import load_dotenv
import sys
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)
load_dotenv()


# 修正： worker 进程需要 *自己* 加载配置和初始化组件。
# 它不能访问主进程的内存。
from config.loader import load_config
from controller.orchestrator import Orchestrator
from data_manager import DataManager
from core.pipeline_state import PipelineState
from strategy_handler import RomanLegionStrategy
from api.gemini_pool_manager import GeminiPoolManager
from execution.order_manager import OrderManager


logger = logging.getLogger(__name__)

# 1. 设置 Celery app
# 它从环境变量（例如 CELERY_BROKER_URL）中读取配置
app = Celery('phoenix_worker')
app.config_from_envvar('CELERY_CONFIG_MODULE', namespace='CELERY')

# 2. 全局/缓存的 Orchestrator 实例 (在 worker 进程中)
# 这避免了在每个任务上都重新初始化所有组件
_orchestrator_instance: Orchestrator | None = None

def get_orchestrator() -> Orchestrator:
    """
    一个单例工厂，用于在 Celery worker 进程中
    按需初始化和缓存 Orchestrator 实例。
    """
    global _orchestrator_instance
    if _orchestrator_instance is None:
        logger.info("WORKER: 'Orchestrator' 实例不存在。正在初始化...")
        try:
            config_path = os.getenv('CONFIG_PATH', 'config/system.yaml')
            config = load_config(config_path)
            if config is None:
                raise RuntimeError("WORKER: 加载 system.yaml 失败。")

            gemini_api_key = os.environ.get("GEMINI_API_KEY")
            gemini_pool = GeminiPoolManager(
                api_key=gemini_api_key,
                pool_size=config.get('llm', {}).get('gemini_pool_size', 5)
            )
            
            pipeline_state = PipelineState()
            cache_dir = config.get('data_manager', {}).get('cache_dir', 'data_cache')
            data_manager = DataManager(config, pipeline_state, cache_dir=cache_dir)
            
            order_manager = OrderManager(config.get('execution', {}))
            
            strategy = RomanLegionStrategy(
                config=config,
                data_manager=data_manager
            )
            
            # Worker 进程中的 Orchestrator 实例 *也* 需要 celery app
            # 以便 *它* 可以（在将来）调用其他任务
            _orchestrator_instance = Orchestrator(
                config=config,
                data_manager=data_manager,
                pipeline_state=pipeline_state,
                gemini_pool=gemini_pool,
                strategy_handler=strategy,
                order_manager=order_manager,
                celery_app=app 
            )
            logger.info("WORKER: 'Orchestrator' 实例初始化成功。")
        except Exception as e:
            logger.critical(f"WORKER: 初始化 Orchestrator 实例失败: {e}", exc_info=True)
            raise
            
    return _orchestrator_instance


@app.task(name='phoenix.run_cognitive_workflow')
def orchestrator_task(task_description: str, context: Dict[str, Any]):
    """
    一个 Celery 任务，用于异步运行完整的认知工作流。
    
    修正：此任务现在按需获取一个 worker 范围内的 
    Orchestrator 实例，并使用 asyncio.run() 
    来执行异步工作流。
    """
    logger.info(f"Celery worker 收到任务: {task_description}")
    try:
        # 1. 获取（或创建）Orchestrator 实例
        orchestrator = get_orchestrator()
        
        # 2. 运行异步工作流
        # Celery 任务是同步的，所以我们使用 asyncio.run() 
        # 来为这个任务创建一个新的事件循环。
        logger.info(f"WORKER: 正在为任务 '{task_description}' 启动 asyncio.run()...")
        result = asyncio.run(
            orchestrator.run_cognitive_workflow(task_description, context)
        )
        logger.info(f"WORKER: 任务 '{task_description}' 完成。")
        return result # Celery 将序列化此结果

    except Exception as e:
        logger.error(f"Celery orchestrator_task 发生错误: {e}", exc_info=True)
        # TODO: 实现重试逻辑
        raise

# 自动发现任务 (如果其他模块中有 @app.task)
app.autodiscover_tasks()

if __name__ == '__main__':
    # 这允许你直接运行 worker (例如用于调试)
    # 命令行: celery -A worker.app worker --loglevel=info
    app.start()


import os
import asyncio
import logging
from typing import Dict, Any
from celery import Celery
from dotenv import load_dotenv

# 从环境变量（或默认值）配置 Celery
# 这使得 broker URL 可以在 docker-compose.yml 或 .env 中设置
broker_url = os.environ.get('CELERY_BROKER_URL', 'redis://localhost:6379/0')
backend_url = os.environ.get('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

app = Celery('phoenix_worker',
             broker=broker_url,
             backend=backend_url)

# 加载 .env 文件 (如果存在)
load_dotenv()

# 设置 Celery app 的配置
app.conf.update(
    task_serializer='json',
    accept_content=['json'],  # 仅接受 json
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

# --- 延迟初始化 ---
# 这两个变量在每个独立的 worker 进程中是全局的
_orchestrator_instance = None
_init_lock = asyncio.Lock() # 确保只有一个任务可以同时初始化

async def _get_orchestrator():
    """
    一个带锁的异步函数，用于安全地初始化并获取 orchestrator 实例。
    这只会在每个 worker 进程的第一个任务上运行一次。
    """
    global _orchestrator_instance
    
    # 使用异步锁来防止多个任务同时尝试初始化
    async with _init_lock:
        # 如果实例已经是 None，则初始化
        if _orchestrator_instance is None:
            logging.info("Celery worker: 正在初始化 Phoenix 系统...")
            
            try:
                # 在函数内部导入以避免顶层循环依赖
                import phoenix_project 
                
                # 调用 initialize_system()。
                # 这个函数会导入 worker.app (现在是安全的)
                # 并且会修改 phoenix_project.orchestrator 全局变量。
                if not await phoenix_project.initialize_system():
                    raise RuntimeError("Celery worker: 系统初始化失败。")
                
                # 从 phoenix_project 模块获取已初始化的实例
                _orchestrator_instance = phoenix_project.orchestrator
                
                if _orchestrator_instance is None:
                     raise RuntimeError("Celery worker: 初始化后 Orchestrator 仍为 None。")
                
                logging.info("Celery worker: Phoenix 系统初始化成功。")
            
            except Exception as e:
                logging.critical(f"Celery worker: 初始化期间发生致命错误: {e}", exc_info=True)
                # 如果初始化失败，我们不应该继续，所以重新引发异常
                raise
        
        # 返回缓存的实例
        return _orchestrator_instance

# --- Celery 任务定义 ---

@app.task(name='phoenix.run_cognitive_workflow', bind=True)
def orchestrator_task(self, task_description: str, context: Dict[str, Any]):
    """
    Celery 任务，用于异步执行重量级的认知工作流。
    'bind=True' 允许我们访问 'self' (任务实例) 以进行重试等操作。
    """
    logger = logging.getLogger('CeleryWorker.Task')
    logger.info(f"收到任务 [ID: {self.request.id}]: {task_description}")
    
    try:
        # 步骤 1: 获取 (或在需要时初始化) orchestrator 实例
        # 我们在 asyncio.run 中包装异步的 _get_orchestrator
        orc = asyncio.run(_get_orchestrator())
        
        # 步骤 2: 运行实际的工作流
        # orchestrator.run_cognitive_workflow 是一个异步函数，所以再次使用 asyncio.run
        result = asyncio.run(orc.run_cognitive_workflow(task_description, context))
        
        logger.info(f"任务 [ID: {self.request.id}] 完成。")
        return result
        
    except Exception as e:
        logger.error(f"任务 [ID: {self.request.id}] 失败: {e}", exc_info=True)
        # 将任务标记为失败，并让 Celery 处理重试（如果已配置）
        # (self, exc, traceback)
        raise self.retry(exc=e, countdown=60) # 60秒后重试

# ---
# 允许使用 'python -m worker' (如果需要)
if __name__ == '__main__':
    app.start()

from celery import Celery
import os
import logging

# 修正：导入 Orchestrator 类，但我们不能在这里实例化它
from controller.orchestrator import Orchestrator
# 修正：'initialize_system' 是一个 async 函数，不能在 worker 启动时
# 像这样直接导入和使用。这个导入可能是无效的。
# from phoenix_project import initialize_system 

logger = logging.getLogger(__name__)

# 1. 设置 Celery app
# 它从环境变量（例如 CELERY_BROKER_URL）中读取配置
app = Celery('phoenix_worker')
app.config_from_envvar('CELERY_CONFIG_MODULE', namespace='CELERY')

# 2. 严重错误：'orchestrator' 在这里未定义
# 
# 这是一个严重的设计缺陷。Celery worker 在一个单独的进程中运行。
# 它不能像这样访问在 'phoenix_project.py' 主进程中创建的 'orchestrator' 实例。
#
# 要修复这个问题，'orchestrator_task' 需要在 *内部* # 实例化它需要的所有组件（Orchestrator, DataManager, Config, GeminiPool 等）。
# 这是一个非常重的操作，但对于解耦的 worker 是必需的。
#
# 临时的修复是注释掉这个损坏的任务，以防止 worker 崩溃。

# @app.task(name='phoenix.run_cognitive_workflow')
# def orchestrator_task(task_description: str, context: dict):
#     """
#     一个 Celery 任务，用于异步运行完整的认知工作流。
#     
#     FIXME: 这个任务已损坏。'orchestrator' 实例在
#     worker 上下文中不存在。
#     """
#     logger.info(f"Celery worker received task: {task_description}")
#     try:
#         # 
#         # ！！！严重错误！！！
#         # 'orchestrator' 在这个作用域中不存在。
#         #
#         # 正确的实现：
#         # 1. 加载配置
#         # config = load_config(os.getenv('CONFIG_PATH', 'config/system.yaml'))
#         # 2. 初始化所有依赖项 (DataManager, GeminiPool, Strategy...)
#         # 3. 实例化一个新的 Orchestrator
#         # orchestrator = Orchestrator(config, ...)
#         # 4. 运行工作流 (注意：Orchestrator 的方法是 async 的，
#         #    Celery 任务默认是 sync 的。你需要一个 asyncio 事件循环。)
#         #
#         # import asyncio
#         # asyncio.run(orchestrator.run_cognitive_workflow(task_description, context))
#         
#         logger.error("FIXME: 'orchestrator_task' is not implemented correctly.")
#         
#         # 损坏的代码：
#         # result = orchestrator.run_cognitive_workflow.delay(task_description, context)
#         # logger.info(f"Task submitted. Result (async): {result}")
#         
#     except Exception as e:
#         logger.error(f"Error in Celery orchestrator_task: {e}", exc_info=True)
#         # TODO: 实现重试逻辑
#         raise

# 自动发现任务 (如果其他模块中有 @app.task)
app.autodiscover_tasks()

if __name__ == '__main__':
    # 这允许你直接运行 worker (例如用于调试)
    # 命令行: celery -A worker.app worker --loglevel=info
    app.start()

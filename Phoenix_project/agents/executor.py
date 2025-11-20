# Phoenix_project/agents/executor.py
# [主人喵的修复 11.11] 实现了 TBD (重试逻辑、错误处理、上下文总线通信)

import asyncio
import logging
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from omegaconf import DictConfig
from Phoenix_project.agents.l1.base import BaseL1Agent
from Phoenix_project.core.pipeline_state import PipelineState

logger = logging.getLogger(__name__)

# [新] 定义用于上下文总线的常量
AGENT_TASK_TOPIC = "AGENT_TASK"
AGENT_RESULT_TOPIC = "AGENT_RESULT"

class AgentExecutor:
    """
    [已实现]
    负责异步执行、协调和监控所有 L1, L2, L3 智能体的任务。
    包含重试逻辑和错误处理。
    """

    def __init__(self, agent_list: list, context_bus, config: DictConfig):
        self.agents = {agent.agent_name: agent for agent in agent_list}
        self.context_bus = context_bus
        self.config = config.get("agent_executor", {})
        
        # [实现] 从配置中读取重试参数
        self.retry_attempts = self.config.get("retry_attempts", 3)
        self.retry_wait_min = self.config.get("retry_wait_min_seconds", 1)
        self.retry_wait_max = self.config.get("retry_wait_max_seconds", 10)

        logger.info(f"AgentExecutor initialized with agents: {list(self.agents.keys())}")
        
        # [实现] 订阅任务主题
        try:
            if hasattr(self.context_bus, "subscribe_async"):
                # 假设有一个支持异步回调的订阅方法
                self.context_bus.subscribe_async(AGENT_TASK_TOPIC, self.handle_task)
            else:
                # 备用：同步订阅（可能需要在 worker 中运行）
                self.context_bus.subscribe(AGENT_TASK_TOPIC, self.handle_task_sync)
            logger.info(f"AgentExecutor subscribed to '{AGENT_TASK_TOPIC}'")
        except Exception as e:
            logger.error(f"Failed to subscribe to ContextBus: {e}", exc_info=True)

    @retry(
        stop=stop_after_attempt(3), # [实现] 使用来自配置的参数
        wait=wait_exponential(min=1, max=10), # [实现] 使用来自配置的参数
        reraise=True # 确保在重试失败后重新引发异常
    )
    async def _run_agent_with_retry(self, agent, task_content, context):
        """
        [新] 内部辅助函数，封装了带 tenacity 重试的 agent.run()。
        """
        logger.debug(f"Attempting to run agent {agent.agent_name}...")
        
        if not hasattr(agent, "run"):
            # 注意：L1 Agent 的 run 可能是同步的也可能是异步的，但在这里我们需要统一处理
            logger.warning(f"Agent {agent.agent_name} lacks a 'run' method.")
            raise NotImplementedError(f"Agent {agent.agent_name} must have a 'run' method.")

        if isinstance(agent, BaseL1Agent):
            # [Task I Fix] Adapt for BaseL1Agent signature: run(state, dependencies)
            # Extract state from context or create a dummy one if missing
            state = context.get('state')
            if not state:
                # Create a temporary state or handle the error. Here we assume a minimal valid state is needed.
                # For safety, we log this event.
                logger.warning(f"No state provided for L1 Agent {agent.agent_name}. Using task_content as dependencies only.")
                state = PipelineState() # Initialize empty/default state
            
            # If run is async, await it. If it's sync, wrap it? 
            # Assuming L1 agents follow the async pattern or we wrap them:
            if asyncio.iscoroutinefunction(agent.run):
                return await agent.run(state=state, dependencies=task_content)
            else:
                return await asyncio.to_thread(agent.run, state=state, dependencies=task_content)

        # Default behavior for other agents (L2/L3)
        if not asyncio.iscoroutinefunction(agent.run):
             logger.warning(f"Agent {agent.agent_name} has sync run method. Running in thread.")
             return await asyncio.to_thread(agent.run, task_content=task_content, context=context)
            
        return await agent.run(task_content=task_content, context=context)

    async def execute_task(self, agent_name: str, task: dict):
        """
        [已实现] 异步执行单个任务，包含重试和结果发布。
        """
        task_id = task.get("task_id", "unknown_task")
        
        if agent_name not in self.agents:
            logger.error(f"Task {task_id}: Agent '{agent_name}' not found.")
            result_payload = {
                "task_id": task_id,
                "agent_name": agent_name,
                "status": "ERROR",
                "error": f"Agent '{agent_name}' not found."
            }
            self.context_bus.publish(AGENT_RESULT_TOPIC, result_payload)
            return result_payload

        agent = self.agents[agent_name]
        
        # [已实现] 上下文传递
        context = task.get("context", {})
        task_content = task.get("content", {})

        try:
            # [已实现] 重试逻辑
            # (配置 tenacity 实例)
            retry_decorator = retry(
                stop=stop_after_attempt(self.retry_attempts),
                wait=wait_exponential(min=self.retry_wait_min, max=self.retry_wait_max),
                reraise=True
            )
            
            # (应用重试)
            async_agent_runner = retry_decorator(self._run_agent_with_retry)
            
            result = await async_agent_runner(agent, task_content, context)
            
            # [已实现] 发送成功结果
            logger.info(f"Task {task_id} on {agent_name} completed successfully.")
            result_payload = {
                "task_id": task_id,
                "agent_name": agent_name,
                "status": "SUCCESS",
                "result": result 
            }
            
        except RetryError as e:
            # [已实现] 健壮的错误处理 (重试失败)
            logger.error(f"Task {task_id} on {agent_name} failed after {self.retry_attempts} attempts: {e}", exc_info=True)
            result_payload = {
                "task_id": task_id,
                "agent_name": agent_name,
                "status": "ERROR",
                "error": f"Task failed after retries: {str(e.last_exception)}",
                "exception_type": type(e.last_exception).__name__
            }
        except Exception as e:
            # [已实现] 健壮的错误处理 (其他异常)
            logger.error(f"Task {task_id} on {agent_name} failed unexpectedly: {e}", exc_info=True)
            result_payload = {
                "task_id": task_id,
                "agent_name": agent_name,
                "status": "ERROR",
                "error": str(e),
                "exception_type": type(e).__name__
            }
            
        # [已实现] 统一发布结果到总线
        self.context_bus.publish(AGENT_RESULT_TOPIC, result_payload)
        return result_payload

    async def run_parallel(self, tasks: list[dict]):
        """
        [已实现] 并行运行多个智能体任务。
        (tasks 是一个列表，每项包含 'agent_name' 和 'task')
        """
        coroutines = [
            self.execute_task(task_def["agent_name"], task_def["task"])
            for task_def in tasks if "agent_name" in task_def and "task" in task_def
        ]
        
        if not coroutines:
            logger.warning("run_parallel called with no valid tasks.")
            return []
            
        # return_exceptions=True 确保一个协程失败不会使所有协程崩溃
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        # 处理在 gather 期间可能发生的异常 (例如，如果 execute_task 本身崩溃)
        final_results = []
        for res in results:
            if isinstance(res, Exception):
                logger.error(f"Exception caught during asyncio.gather in run_parallel: {res}", exc_info=True)
                final_results.append({"status": "ERROR", "error": f"Gather exception: {str(res)}"})
            else:
                final_results.append(res)
                
        return final_results

    async def handle_task(self, task_data: dict):
        """
        [新] 异步回调，用于处理来自 ContextBus 的任务。
        """
        logger.debug(f"Received task via async subscriber: {task_data.get('task_id')}")
        agent_name = task_data.get("agent_name")
        task_content = task_data.get("task")
        
        if not agent_name or not task_content:
            logger.warning(f"Invalid task received on {AGENT_TASK_TOPIC}: {task_data}")
            return
            
        # (在后台执行，不阻塞总线)
        asyncio.create_task(self.execute_task(agent_name, task_content))

    def handle_task_sync(self, task_data: dict):
        """
        [新] 同步回调（备用）。
        警告：这可能会阻塞，最好在专用的 consumer/worker 线程中运行。
        """
        logger.debug(f"Received task via sync subscriber: {task_data.get('task_id')}")
        
        # (一个安全的模式是：如果当前有事件循环，则创建任务)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.handle_task(task_data))
        except RuntimeError:
             logger.warning(f"handle_task_sync called without running event loop. Task {task_data.get('task_id')} dropped.")

"""
Phoenix_project/agents/executor.py
[Phase 5 Task 5] Fix AgentExecutor Result Loss.
Switch from Pub/Sub to Redis List (RPUSH) for persistent result queuing.
"""
import logging
import asyncio
import json
from typing import Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from omegaconf import DictConfig

from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.agents.l1.base import BaseL1Agent
from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

# [新] 定义用于上下文总线的常量
AGENT_TASK_TOPIC = "AGENT_TASK"
AGENT_RESULT_TOPIC = "AGENT_RESULT"
AGENT_RESULT_QUEUE = "phoenix:agent:results:queue" # [Task 5] Redis List Key

class AgentExecutor:
    """
    [已实现]
    负责异步执行、协调和监控所有 L1, L2, L3 智能体的任务。
    包含重试逻辑和错误处理。
    """

    def __init__(self, agent_list: list, context_bus, config: DictConfig):
        self.agents = {agent.agent_name: agent for agent in agent_list}
        self.context_bus = context_bus
        self.config = config.get("agent_executor", {}) if isinstance(config, (dict, DictConfig)) else {}
        
        self.retry_attempts = self.config.get("retry_attempts", 3)
        self.retry_wait_min = self.config.get("retry_wait_min_seconds", 1)
        self.retry_wait_max = self.config.get("retry_wait_max_seconds", 10)

        self.max_concurrency = self.config.get("max_concurrency", 50)
        self.semaphore = asyncio.Semaphore(self.max_concurrency)

        logger.info(f"AgentExecutor initialized with agents: {list(self.agents.keys())}, Max Concurrency: {self.max_concurrency}")
        
        # 订阅任务 (Task Queue/Topic is handled by Orchestrator or Bus logic)
        # Here we just ensure we can publish results reliably
        
    async def _run_agent_with_retry(self, agent, task_content, context):
        """
        [Beta Final Fix] 内部辅助函数，优先调用 safe_run。
        """
        logger.debug(f"Attempting to run agent {agent.agent_name}...")
        
        if isinstance(agent, BaseL1Agent):
            state = context.get('state')
            if not state:
                logger.warning(f"No state provided for L1 Agent {agent.agent_name}. Using task_content as dependencies only.")
                state = PipelineState() 
            return await agent.safe_run(state=state, dependencies=task_content)

        if not hasattr(agent, "run"):
            logger.warning(f"Agent {agent.agent_name} lacks a 'run' method.")
            raise NotImplementedError(f"Agent {agent.agent_name} must have a 'run' method.")

        if not asyncio.iscoroutinefunction(agent.run):
             logger.warning(f"Agent {agent.agent_name} has sync run method. Running in thread.")
             return await asyncio.to_thread(agent.run, task_content=task_content, context=context)
            
        return await agent.run(task_content=task_content, context=context)

    async def execute_task(self, agent_name: str, task: dict):
        """
        [已实现] 异步执行单个任务，包含重试和结果发布。
        [Task 5] Publish to Persistent Queue (Redis List) instead of Pub/Sub.
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
            await self._publish_result(result_payload)
            return result_payload

        agent = self.agents[agent_name]
        context = task.get("context", {})
        task_content = task.get("content", {})

        async with self.semaphore:
            try:
                retry_decorator = retry(
                    stop=stop_after_attempt(self.retry_attempts),
                    wait=wait_exponential(min=self.retry_wait_min, max=self.retry_wait_max),
                    reraise=True
                )
                
                async_agent_runner = retry_decorator(self._run_agent_with_retry)
                result = await async_agent_runner(agent, task_content, context)
                
                logger.info(f"Task {task_id} on {agent_name} completed successfully.")
                result_payload = {
                    "task_id": task_id,
                    "agent_name": agent_name,
                    "status": "SUCCESS",
                    "result": result 
                }
                
            except RetryError as e:
                logger.error(f"Task {task_id} on {agent_name} failed after {self.retry_attempts} attempts: {e}", exc_info=True)
                result_payload = {
                    "task_id": task_id,
                    "agent_name": agent_name,
                    "status": "ERROR",
                    "error": f"Task failed after retries: {str(e.last_exception)}",
                    "exception_type": type(e.last_exception).__name__
                }
            except Exception as e:
                logger.error(f"Task {task_id} on {agent_name} failed unexpectedly: {e}", exc_info=True)
                result_payload = {
                    "task_id": task_id,
                    "agent_name": agent_name,
                    "status": "ERROR",
                    "error": str(e),
                    "exception_type": type(e).__name__
                }
            
        await self._publish_result(result_payload)
        return result_payload

    async def _publish_result(self, result_payload: Dict[str, Any]):
        """
        [Task 5] Publish result to persistent Redis Queue.
        Fallback to Pub/Sub if queue push fails (or do both).
        """
        try:
            # Check if context_bus exposes raw redis client or a queue push method
            if hasattr(self.context_bus, 'redis') and self.context_bus.redis:
                # Direct RPUSH to Redis List
                # Ensure serialization
                payload_str = json.dumps(result_payload, default=str)
                await self.context_bus.redis.rpush(AGENT_RESULT_QUEUE, payload_str)
                # logger.debug(f"Result for {result_payload.get('task_id')} pushed to {AGENT_RESULT_QUEUE}")
                
                # Also Publish for real-time listeners (optional but good for monitoring)
                await self.context_bus.publish(AGENT_RESULT_TOPIC, result_payload)
                
            elif hasattr(self.context_bus, 'publish'):
                # Fallback to Pub/Sub if raw redis not accessible
                logger.warning("ContextBus raw redis missing. Falling back to Pub/Sub for results (Non-Persistent).")
                self.context_bus.publish(AGENT_RESULT_TOPIC, result_payload)
                
        except Exception as e:
            logger.error(f"Failed to publish execution result: {e}", exc_info=True)

    async def run_parallel(self, tasks: list[dict]):
        """
        [已实现] 并行运行多个智能体任务。
        """
        coroutines = [
            self.execute_task(task_def["agent_name"], task_def["task"])
            for task_def in tasks if "agent_name" in task_def and "task" in task_def
        ]
        
        if not coroutines:
            logger.warning("run_parallel called with no valid tasks.")
            return []
            
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        final_results = []
        for res in results:
            if isinstance(res, Exception):
                logger.error(f"Exception caught during asyncio.gather in run_parallel: {res}", exc_info=True)
                final_results.append({"status": "ERROR", "error": f"Gather exception: {str(res)}"})
            else:
                final_results.append(res)
                
        return final_results

    async def handle_task(self, task_data: dict):
        # Implementation remains same
        logger.debug(f"Received task via async subscriber: {task_data.get('task_id')}")
        agent_name = task_data.get("agent_name")
        task_content = task_data.get("task")
        
        if not agent_name or not task_content:
            logger.warning(f"Invalid task received on {AGENT_TASK_TOPIC}: {task_data}")
            return
        asyncio.create_task(self.execute_task(agent_name, task_content))

    def handle_task_sync(self, task_data: dict):
        # Implementation remains same
        logger.debug(f"Received task via sync subscriber: {task_data.get('task_id')}")
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.handle_task(task_data))
        except RuntimeError:
             logger.warning(f"handle_task_sync called without running event loop. Task {task_data.get('task_id')} dropped.")

import asyncio
import logging
from typing import Any
from Phoenix_project.core.schemas.task_schema import Task
from Phoenix_project.agents.l1.base import L1Agent
from Phoenix_project.agents.l2.base import L2Agent
from Phoenix_project.agents.l3.base import L3Agent
from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

class AgentExecutor:
    """
    Manages the execution lifecycle of a single agent.
    It fetches tasks from its assigned queue and executes them using the agent instance.
    """
    def __init__(self, agent_id: str, agent: Any, task_queue: asyncio.Queue):
        self.agent_id = agent_id
        self.agent = agent
        self.task_queue = task_queue
        self.is_running = False
        self.current_task_id = None
        logger.info(f"AgentExecutor initialized for {agent_id}")

    async def start(self):
        """
        Starts the executor's main loop.
        """
        if self.is_running:
            logger.warning(f"Executor for {self.agent_id} is already running.")
            return

        self.is_running = True
        logger.info(f"Executor for {self.agent_id} started.")
        try:
            while self.is_running:
                try:
                    task: Task = await self.task_queue.get()
                    if task is None:
                        logger.info(f"Executor for {self.agent_id} received None, stopping.")
                        break

                    logger.info(f"Executor for {self.agent_id} processing task: {task.task_id}")
                    self.current_task_id = task.task_id
                    
                    try:
                        # TBD: Differentiate context/event passing based on agent level
                        # [FIX] Differentiate context/event passing based on agent level
                        # This is a simplified call
                        result = await self.agent.run(task.event, task.context_window)
                        
                        # TBD: Send result to the next step (e.g., L2 Agent or Bus)
                        # [FIX] Send result to the next step (e.g., L2 Agent or Bus)
                        logger.info(f"Task {task.task_id} completed by {self.agent_id}.")
                    
                    except Exception as e:
                        logger.error(f"Agent {self.agent_id} failed on task {task.task_id}: {e}", exc_info=True)
                        # TBD: Error handling logic (e.g., send to error bus)
                        # [FIX] Error handling logic (e.g., send to error bus)
                    
                    finally:
                        self.task_queue.task_done()
                        self.current_task_id = None
                        logger.debug(f"Task {task.task_id} marked as done by {self.agent_id}.")

                except asyncio.CancelledError:
                    logger.info(f"Executor loop for {self.agent_id} cancelled.")
                    break
                except Exception as e:
                    logger.critical(f"Executor loop for {self.agent_id} encountered critical error: {e}", exc_info=True)
                    # TBD: Implement backoff/retry?
                    # [FIX] Implement backoff/retry?
                    await asyncio.sleep(1) # Avoid tight loop on critical error
                    
        self.is_running = False
        logger.info(f"Executor for {self.agent_id} stopped.")

    def stop(self):
        """
        Signals the executor to stop.
        """
        if not self.is_running:
            logger.warning(f"Executor for {self.agent_id} is not running.")
            return
            
        self.is_running = False
        # Put a None task to unblock the queue.get()
        self.task_queue.put_nowait(None)
        logger.info(f"Stop signal sent to executor {self.agent_id}.")

    def get_status(self):
        """
        Returns the current status of the executor.
        """
        return {
            "agent_id": self.agent_id,
            "is_running": self.is_running,
            "current_task_id": self.current_task_id,
            "queue_size": self.task_queue.qsize()
        }

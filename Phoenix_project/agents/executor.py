import asyncio
from typing import Dict, Any
import pandas as pd # [任务 C.1] 导入 pandas
from Phoenix_project.agents.l1.base import L1BaseAgent
from Phoenix_project.agents.l2.base import L2BaseAgent
from Phoenix_project.agents.l3.base import L3BaseAgent
from Phoenix_project.core.schemas.task_schema import AgentTask
from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

class Executor:
    """
    Manages the execution queue and lifecycle for a specific agent.
    """
    
    def __init__(
        self, 
        agent_id: str, 
        agent_registry: Dict[str, Any], 
        max_queue_size: int = 100
    ):
        """
        Initializes the Executor.
        
        Args:
            agent_id: The ID of the agent this executor manages.
            agent_registry: A dictionary mapping agent IDs to agent instances.
            max_queue_size: Maximum tasks allowed in the queue.
        """
        self.agent_id = agent_id
        self.agent_registry = agent_registry
        self.task_queue = asyncio.Queue(maxsize=max_queue_size)
        self.is_running = False
        self._task = None
        
        # [任务 C.1] 丰富的状态跟踪
        self.current_task_id: Optional[str] = None
        self.last_processed_time: Optional[pd.Timestamp] = None
        
        logger.info(f"Executor for {agent_id} initialized.")

    async def start(self):
        """
        Starts the executor's processing loop.
        """
        if self.is_running:
            logger.warning(f"Executor for {agent_id} is already running.")
            return
            
        self.is_running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(f"Executor for {agent_id} started.")

    async def stop(self):
        """
        Stops the executor's processing loop.
        """
        if not self.is_running:
            logger.info(f"Executor for {agent_id} is not running.")
            return
            
        self.is_running = False
        if self._task:
            self.task_queue.put_nowait(None) # Sentinel value to stop loop
            try:
                await self._task
            except asyncio.CancelledError:
                logger.info(f"Executor task for {agent_id} was cancelled.")
        logger.info(f"Executor for {agent_id} stopped.")

    async def submit_task(self, task: AgentTask) -> bool:
        """
        Submits a new task to the agent's queue.
        
        Args:
            task: The AgentTask to be executed.
            
        Returns:
            True if submitted, False if the queue is full.
        """
        if not self.is_running:
            logger.error(f"Cannot submit task: Executor {self.agent_id} is not running.")
            return False
            
        try:
            self.task_queue.put_nowait(task)
            logger.debug(f"Task {task.task_id} submitted to {self.agent_id}")
            return True
        except asyncio.QueueFull:
            logger.warning(f"Queue full for agent {self.agent_id}. Task {task.task_id} rejected.")
            return False

    async def _run_loop(self):
        """
        The main processing loop for the agent.
        """
        agent = self.agent_registry.get(self.agent_id)
        if not agent:
            logger.critical(f"Agent {self.agent_id} not found in registry. Executor shutting down.")
            self.is_running = False
            return

        while self.is_running:
            try:
                task = await self.task_queue.get()
                
                if task is None: # Sentinel value
                    logger.info(f"Shutdown signal received by {self.agent_id}.")
                    break
                
                logger.info(f"Executor {self.agent_id} processing task {task.task_id}...")
                
                # [任务 C.1] 更新丰富的状态
                self.current_task_id = task.task_id
                
                try:
                    # TBD: Differentiate context/event passing based on agent level
                    # This is a simplified call
                    result = await agent.run(task.event, task.context_window)
                    
                    # TBD: Send result to the next step (e.g., L2 Agent or Bus)
                    logger.info(f"Task {task.task_id} completed by {self.agent_id}.")
                
                except Exception as e:
                    logger.error(f"Agent {self.agent_id} failed on task {task.task_id}: {e}", exc_info=True)
                    # TBD: Error handling logic (e.g., send to error bus)
                
                finally:
                    self.task_queue.task_done()
                    # [任务 C.1] 更新丰富的状态
                    self.current_task_id = None
                    self.last_processed_time = pd.Timestamp.now(tz='UTC')

            except asyncio.CancelledError:
                logger.info(f"Run loop for {self.agent_id} cancelled.")
                break
            except Exception as e:
                logger.critical(f"Executor loop for {self.agent_id} encountered critical error: {e}", exc_info=True)
                # TBD: Implement backoff/retry?
                await asyncio.sleep(1) # Avoid tight loop on critical error
                
        self.is_running = False
        logger.info(f"Executor loop for {self.agent_id} has exited.")

    def get_agent_status(self) -> Dict[str, Any]:
        """
        Returns the current status of the agent.
        
        [任务 C.1] TODO: Implement richer status tracking.
        """
        # [任务 C.1] 已实现
        return {
            "agent_id": self.agent_id,
            "status": "IDLE" if self.current_task_id is None else "BUSY",
            "queue_depth": self.task_queue.qsize(),
            "current_task_id": self.current_task_id,
            "last_processed_time": str(self.last_processed_time) if self.last_processed_time else None
        }

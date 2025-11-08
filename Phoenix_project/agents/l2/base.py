import logging
from abc import ABC, abstractmethod
from typing import List, Any, AsyncGenerator

# 确保导入 Task 类型（即使是作为类型提示）
try:
    from Phoenix_project.core.schemas.task_schema import Task
except ImportError:
    Task = Any # Fallback

logger = logging.getLogger(__name__)

class L2Agent(ABC):
    """
    L2 智能体基类（元认知层）。
    这些智能体通常依赖于 L1 智能体的输出。
    """
    def __init__(
        self,
        agent_id: str,
        llm_client: Any,
        data_manager: Any,
    ):
        """
        初始化 L2 智能体。
        
        参数:
            agent_id (str): 智能体的唯一标识符。
            llm_client (Any): 用于与 LLM API 交互的客户端。
            data_manager (Any): 用于检索数据的 DataManager (L2 可能也需要)。
        """
        self.agent_id = agent_id
        self.llm_client = llm_client
        self.data_manager = data_manager
        logger.info(f"L2 Agent {self.agent_id} (Type: {type(self).__name__}) initialized.")

    @abstractmethod
    async def run(self, task: "Task", dependencies: List[Any]) -> AsyncGenerator[Any, None]:
        """
        异步运行智能体。
        L2 智能体接收 'dependencies' (来自 L1 的输出) 而不是 'context'。
        
        参数:
            task (Task): 当前任务。
            dependencies (List[Any]): 来自上游智能体（L1）的输出列表。
            
        收益:
            AsyncGenerator[Any, None]: 异步生成结果（例如 EvidenceItem, CriticResult）。
        """
        raise NotImplementedError
        yield

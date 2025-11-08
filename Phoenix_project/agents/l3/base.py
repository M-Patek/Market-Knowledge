import logging
from abc import ABC, abstractmethod
from typing import List, Any, AsyncGenerator

# 确保导入 Task 类型（即使是作为类型提示）
try:
    from Phoenix_project.core.schemas.task_schema import Task
except ImportError:
    Task = Any # Fallback

logger = logging.getLogger(__name__)

class L3Agent(ABC):
    """
    L3 智能体基类（决策与执行转换层）。
    这些智能体（例如 DRL/Quant）将 L2 的决策（例如 FusionResult）
    转换为 L4 执行层可以理解的信号（例如 Signal）。
    """
    def __init__(
        self, 
        agent_id: str,
        model_client: Any,  # DRL/Quant/Quant 模型的客户端
        data_manager: Any   # 用于获取实时状态/特征
    ):
        """
        初始化 L3 智能体。
        
        参数:
            agent_id (str): 智能体的唯一标识符。
            model_client (Any): 用于加载和运行 DRL/Quant 模型的客户端。
            data_manager (Any): 用于检索数据（例如：实时价格、波动率）。
        """
        self.agent_id = agent_id
        self.model_client = model_client
        self.data_manager = data_manager
        logger.info(f"L3 Agent {self.agent_id} (Type: {type(self).__name__}) initialized.")

    @abstractmethod
    async def run(self, task: "Task", dependencies: List[Any]) -> AsyncGenerator[Any, None]:
        """
        异步运行智能体。
        L3 智能体接收 'dependencies' (来自 L2 的输出)
        
        参数:
            task (Task): 当前任务。
            dependencies (List[Any]): 来自上游智能体（L2）的输出列表。
            
        收益:
            AsyncGenerator[Any, None]: 异步生成结果（例如 Signal）。
        """
        raise NotImplementedError
        yield

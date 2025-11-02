# (原: ai/base_trainer.py)
from abc import ABC, abstractmethod
from typing import Dict, Any
# 修复：将相对导入 'from ..core.pipeline_state...' 更改为绝对导入
from core.pipeline_state import PipelineState
# 修复：将相对导入 'from ..data_manager...' 更改为绝对导入
from data_manager import DataManager
# 修复：将相对导入 'from ..monitor.logging...' 更改为绝对导入
from monitor.logging import get_logger

logger = get_logger(__name__)

class BaseTrainer(ABC):
    """
    训练器的抽象基类。
    定义了所有训练器（如滚动优化、DRL）必须实现的通用接口。
    """

    def __init__(self, config: Dict[str, Any], data_manager: DataManager):
        """
        初始化训练器。

        Args:
            config (Dict[str, Any]): 'training' 部分的配置。
            data_manager (DataManager): 用于访问历史数据的数据管理器。
        """
        self.config = config
        self.data_manager = data_manager
        self.model = None # 训练产出的模型
        logger.info(f"Trainer '{self.__class__.__name__}' 已初始化。")

    @abstractmethod
    async def run_training_loop(self):
        """
        执行主要的训练循环（例如，滚动窗口）。
        """
        pass

    @abstractmethod
    def evaluate_model(self, validation_data: Any) -> Dict[str, float]:
        """
        在验证集上评估当前模型。
        """
        pass

    @abstractmethod
    def save_model(self, path: str):
        """
        将训练好的模型保存到磁盘。
        """
        if self.model is None:
            logger.warning("没有可保存的模型。")
            return
        logger.info(f"模型已保存到: {path}")

    @abstractmethod
    def load_model(self, path: str):
        """
        从磁盘加载模型。
        """
        logger.info(f"模型已从: {path} 加载。")

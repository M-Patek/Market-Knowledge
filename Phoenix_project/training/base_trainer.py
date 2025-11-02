from typing import Dict, Any

class BaseTrainer:
    """
    (新文件)
    所有训练器 (例如 WalkForwardTrainer) 的抽象基类。
    
    这是为了修复 ai/walk_forward_trainer.py 中
    试图导入一个不存在的 .base_trainer 的问题。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化基础训练器。
        
        Args:
            config (Dict[str, Any]): 该训练器的特定配置块。
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"{self.__class__.__name__} initialized.")

    def train(self, data: Any) -> Any:
        """
        在给定的数据上训练模型。
        """
        raise NotImplementedError("Trainer 必须实现 train() 方法")

    def evaluate(self, model_artifacts: Any, validation_data: Any) -> Dict[str, float]:
        """
        在验证数据上评估一个已训练的模型。
        """
        raise NotImplementedError("Trainer 必须实现 evaluate() 方法")

# (原: drl/agents/base_agent.py)
# (无内部导入，无需修复)

from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseAgent(ABC):
    """
    (在线推理) DRL 智能体的基类。
    定义了在“在线推理”阶段（在 CognitiveEngine 中）如何调用 DRL 智能体。
    """
    def __init__(self, config: Dict[str, Any], model_path: str):
        self.config = config
        self.model_path = model_path
        self.model = self.load_model(model_path)

    @abstractmethod
    def load_model(self, path: str) -> Any:
        """
        加载训练好的 DRL 模型 (例如 PPO.load)。
        """
        # 示例:
        # from stable_baselines3 import PPO
        # return PPO.load(path)
        print(f"模拟：从 {path} 加载模型")
        return "loaded_model_object"

    @abstractmethod
    def predict(self, observation: Any) -> Any:
        """
        执行模型推理。
        """
        pass

# (原: drl/drl_model_registry.py)
# 这是一个用于“在线推理”的模块，用于加载“离线训练”产出的模型。

from stable_baselines3 import PPO, SAC, A2C
from typing import Dict, Any
import os

# --- [修复] ---
# 修复：将相对导入 'from ..monitor.logging...' 更改为绝对导入
from Phoenix_project.monitor.logging import get_logger
# --- [修复结束] ---

logger = get_logger(__name__)

class ModelRegistry:
    """
    (在线推理)
    负责加载、管理和提供所有在“离线训练”中训练好的 DRL 模型。
    这个注册表会被 'CognitiveEngine' 或 'MetacognitiveAgent' 用来
    按需获取智能体实例。
    """
    
    SUPPORTED_ALGOS = {
        "PPO": PPO,
        "SAC": SAC,
        "A2C": A2C
    }

    def __init__(self, config: Dict[str, Any], model_base_path: str = "models/"):
        self.config = config.get('model_registry', {})
        self.model_base_path = model_base_path
        self.agents = {} # 缓存加载的智能体模型
        
        logger.info(f"ModelRegistry (模型注册表) 已初始化。模型路径: {model_base_path}")
        
    def load_all_agents(self):
        """
        根据配置加载所有必需的 DRL 智能体。
        """
        agent_configs = self.config.get('agents', [])
        if not agent_configs:
            logger.warning("模型注册表：配置中没有找到 'agents'。")
            return

        logger.info(f"正在加载 {len(agent_configs)} 个 DRL 智能体...")
        
        for agent_conf in agent_configs:
            try:
                name = agent_conf['name']
                algo = agent_conf['algorithm']
                filename = agent_conf['filename']
                
                if algo not in self.SUPPORTED_ALGOS:
                    logger.error(f"不支持的算法: {algo} (用于 {name})。")
                    continue
                    
                model_path = os.path.join(self.model_base_path, filename)
                
                if not os.path.exists(model_path):
                    logger.error(f"模型文件未找到: {model_path} (用于 {name})。")
                    continue
                    
                # 加载 SB3 模型
                model_class = self.SUPPORTED_ALGOS[algo]
                self.agents[name] = model_class.load(model_path)
                logger.info(f"成功加载智能体: '{name}' (算法: {algo})")
                
            except Exception as e:
                logger.error(f"加载智能体 '{agent_conf.get('name')}' 失败: {e}", exc_info=True)

    def get_agent(self, name: str) -> Any:
        """
        获取一个已加载的 DRL 智能体模型。
        """
        model = self.agents.get(name)
        if model is None:
            logger.warning(f"请求的智能体 '{name}' 未在注册表中找到。")
        return model

# 示例:
# registry_config = {
#     "agents": [
#         {"name": "AlphaAgent", "algorithm": "PPO", "filename": "drl_agents_v1_alpha.zip"},
#         {"name": "RiskAgent", "algorithm": "PPO", "filename": "drl_agents_v1_risk.zip"}
#     ]
# }
# registry = ModelRegistry(registry_config)
# registry.load_all_agents()
# alpha_model = registry.get_agent("AlphaAgent")

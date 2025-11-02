# (原: drl/multi_agent_trainer.py)
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Dict, Any

# --- [修复] ---
# 修复：将相对导入 'from .trading_env...' 更改为绝对导入
from training.drl.trading_env import TradingEnv
# 修复：将相对导入 'from .agents.alpha_agent...' 更改为绝对导入
from training.drl.agents.alpha_agent import AlphaAgent
# 修复：将相对导入 'from .agents.risk_agent...' 更改为绝对导入
from training.drl.agents.risk_agent import RiskAgent
# 修复：将相对导入 'from ...utils.replay_buffer...' 更改为绝对导入
from utils.replay_buffer import ReplayBuffer
# 修复：将相对导入 'from ...monitor.logging...' 更改为绝对导入
from monitor.logging import get_logger
# --- [修复结束] ---

logger = get_logger(__name__)

class MultiAgentTrainer:
    """
    负责协调 DRL 智能体（如 AlphaAgent, RiskAgent）的训练。
    使用 TradingEnv 作为模拟环境。
    """

    def __init__(self, config: Dict[str, Any], trading_env: TradingEnv):
        self.config = config.get('drl_training', {})
        self.env = trading_env # DRL 训练环境
        
        # 为环境创建一个 SB3 兼容的 VecEnv 封装
        self.vec_env = DummyVecEnv([lambda: self.env])

        self.total_timesteps = self.config.get('total_timesteps', 1_000_000)
        self.model_save_path = self.config.get('model_save_path', 'models/drl_agents_v1')
        
        # 初始化智能体 (模型)
        # TODO: 这里需要一个多智能体 (MARL) 框架, SB3 本身不支持 MARL。
        # 作为简化，我们假设先训练一个 AlphaAgent。
        # 在一个真实的多智能体设置中，您可能会使用 PettingZoo + Ray RLLib
        
        self.alpha_agent_model = PPO(
            "MlpPolicy",
            self.vec_env,
            verbose=1,
            tensorboard_log="./tensorboard_logs/alpha_agent/"
        )
        
        logger.info(f"MultiAgentTrainer (PPO) 已初始化。将训练 {self.total_timesteps} 步。")

    async def train_agents(self):
        """
        启动 DRL 训练循环。
        """
        logger.info("--- 开始 DRL 智能体训练 ---")
        
        try:
            # 训练 AlphaAgent
            # SB3 的 learn() 是同步的，如果 DRL 训练需要很长时间，
            # 最好在单独的进程中运行它。
            # 为了简单起见，我们在这里直接调用它。
            
            # TODO: 将 learn() 包装在 asyncio.to_thread (Python 3.9+)
            # 或 run_in_executor 中，使其成为非阻塞的。
            
            logger.info(f"开始训练 AlphaAgent...")
            self.alpha_agent_model.learn(
                total_timesteps=self.total_timesteps,
                log_interval=10
            )
            logger.info("AlphaAgent 训练完成。")
            
            # TODO: 在此添加 RiskAgent 和 ExecutionAgent 的训练逻辑
            # ...
            
            logger.info("--- DRL 智能体训练结束 ---")
            self.save_models(self.model_save_path)
            
        except Exception as e:
            logger.error(f"DRL 训练失败: {e}", exc_info=True)

    def save_models(self, path_prefix: str):
        """
        保存所有训练好的智能体模型。
        """
        alpha_path = f"{path_prefix}_alpha.zip"
        self.alpha_agent_model.save(alpha_path)
        logger.info(f"AlphaAgent 模型已保存到: {alpha_path}")
        
        # ... 保存其他智能体 ...

    def load_models(self, path_prefix: str):
        """
        加载智能体模型 (用于在线推理)。
        """
        alpha_path = f"{path_prefix}_alpha.zip"
        self.alpha_agent_model = PPO.load(alpha_path, env=self.vec_env)
        logger.info(f"AlphaAgent 模型已从: {alpha_path} 加载。")

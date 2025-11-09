import asyncio
from typing import Dict, Any

# [MARL 重构] 移除 SB3 导入
# import gymnasium as gym
# from stable_baselines3 import PPO
# from stable_baselines3.common.vec_env import DummyVecEnv

# [MARL 重构] 添加 Ray RLLib 和 PettingZoo 导入
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.policy.policy import PolicySpec

# [MARL 重构] 导入 PettingZoo 版本的 TradingEnv
from Phoenix_project.training.drl.trading_env import TradingEnv

# (L3 智能体导入不再需要，因为模型是在这里训练的)
# from Phoenix_project.agents.l3.alpha_agent import AlphaAgent
# from Phoenix_project.agents.l3.risk_agent import RiskAgent

# (ReplayBuffer 导入不再需要)
# from Phoenix_project.utils.replay_buffer import ReplayBuffer

from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

class MultiAgentTrainer:
    """
    [MARL 重构]
    使用 Ray RLLib 协调 DRL 智能体（AlphaAgent, RiskAgent, ExecutionAgent）的训练。
    """

    def __init__(self, config: Dict[str, Any]):
        """
        [MARL 重构]
        初始化 Ray RLLib 训练器。
        
        Args:
            config (Dict[str, Any]): 包含 'drl_training' 和 'drl_env' 的配置字典。
                                     'drl_env' 必须包含 'data', 'symbol' 等
                                     以传递给 TradingEnv 构造函数。
        """
        self.config = config.get('drl_training', {})
        self.env_config = config.get('drl_env', {}) # RLLib 将使用这个配置
        
        self.total_timesteps = self.config.get('total_timesteps', 1_000_000)
        self.model_save_path = self.config.get('model_save_path', 'models/drl_agents_v1_rllib')
        
        # --- [MARL 重构] 移除 SB3 初始化 ---
        # self.env = trading_env
        # self.vec_env = DummyVecEnv([lambda: self.env])
        # self.alpha_agent_model = PPO(...)

        # --- [MARL 重构] 添加 RLLib 配置 ---
        logger.info("初始化 Ray RLLib...")
        if not ray.is_initialized():
            ray.init(logging_level="ERROR", ignore_reinit_error=True)

        # 1. 注册 PettingZoo 环境
        #    这个 lambda 接收 'config' 字典 (即下面的 env_config)
        #    并将其解包 (**) 传递给 TradingEnv 构造函数。
        register_env("phoenix_marl_env", lambda cfg: TradingEnv(**cfg))

        # 2. 定义策略 (每个智能体可以共享或拥有独立策略)
        #    (我们也可以为所有智能体使用一个共享策略)
        policies = {
            "alpha_policy": PolicySpec(config={"gamma": 0.95}),
            "risk_policy": PolicySpec(config={"gamma": 0.99}),
            "execution_policy": PolicySpec(config={"gamma": 0.9}),
        }

        # 3. 映射智能体到策略
        def policy_mapping_fn(agent_id, episode, worker, **kwargs):
            if agent_id.startswith("alpha_agent"):
                return "alpha_policy"
            elif agent_id.startswith("risk_agent"):
                return "risk_policy"
            else:
                return "execution_policy"

        # 4. 创建 RLLib 算法配置
        algo_config = (
            PPOConfig()
            .environment(
                env="phoenix_marl_env", 
                env_config=self.env_config # 传递环境配置 (包含 'data', 'symbol' 等)
            )
            .framework("torch")
            .resources(num_gpus=0) # 假设没有 GPU，如果可用则设为 1
            .multi_agent(
                policies=policies,
                policy_mapping_fn=policy_mapping_fn
            )
            .training(
                train_batch_size=4000 
            )
            # .rollouts(num_rollout_workers=1) # (可选)
        )

        # 5. 构建算法
        self.algorithm = algo_config.build()
        
        logger.info("Ray RLLib (PPO) Multi-Agent 训练器已初始化。")

    async def train_agents(self):
        """
        [MARL 重构]
        启动 RLLib 训练循环 (非阻塞)。
        解决了 asyncio.to_thread 的 TODO。
        """
        logger.info("--- 开始 RLLib MARL 智能体训练 ---")

        # 定义将在单独线程中运行的阻塞训练函数
        def _blocking_training_loop():
            try:
                # 假设 self.total_timesteps 仍然在 config 中
                timesteps_per_iter = self.algorithm.config.train_batch_size
                if timesteps_per_iter == 0:
                    logger.warning("train_batch_size 为 0，设置为 4000")
                    timesteps_per_iter = 4000
                    
                num_iterations = self.total_timesteps // timesteps_per_iter
                if num_iterations == 0:
                    num_iterations = 1 # 至少运行一次

                logger.info(f"将训练 {num_iterations} 次迭代 (总步数: {self.total_timesteps})")

                for i in range(num_iterations):
                    # RLLib 的 train() 是阻塞的
                    result = self.algorithm.train()

                    # (RLLib 的日志记录很详细，这里可以减少日志频率)
                    if i % 10 == 0:
                        reward_mean = result.get('episode_reward_mean', 'N/A')
                        logger.info(f"迭代 {i}: 平均奖励 = {reward_mean}")
                        
                        # 保存检查点 (RLLib 方式)
                        # RLLib 默认会保存检查点，这里是显式保存
                        self.save_models(self.model_save_path)

                logger.info("--- RLLib MARL 训练结束 ---")

            except Exception as e:
                logger.error(f"RLLib DRL 训练失败: {e}", exc_info=True)

        # [修复] 使用 asyncio.to_thread 运行阻塞循环，避免阻塞主事件循环
        await asyncio.to_thread(_blocking_training_loop)

        # 确保最终模型被保存
        self.save_models(self.model_save_path)


    def save_models(self, path_prefix: str):
        """
        [MARL 重构]
        保存 RLLib 训练器检查点。
        """
        try:
            checkpoint_dir = self.algorithm.save(path_prefix)
            logger.info(f"RLLib 模型检查点已保存到: {checkpoint_dir}")
        except Exception as e:
            logger.error(f"保存 RLLib 模型失败: {e}", exc_info=True)


    def load_models(self, path_prefix: str):
        """
        [MARL 重构]
        从检查点恢复 RLLib 训练器。
        """
        try:
            # RLLib 通常在 build() 时从检查点恢复，或者使用 restore()
            self.algorithm.restore(path_prefix)
            logger.info(f"RLLib 模型已从: {path_prefix} 加载。")
        except Exception as e:
            logger.error(f"加载 RLLib 模型失败: {e}", exc_info=True)

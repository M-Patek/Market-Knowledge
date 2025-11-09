# training/drl/trading_env.py
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

# [任务 1.1] 导入 L2 结果 schema (用于类型提示)
from typing import Dict, Any
from Phoenix_project.core.schemas.fusion_result import FusionResult


class TradingEnv(gym.Env):
    """
    一个用于 DRL 智能体的多资产交易环境 (基于 RLLib 的 gymnasium.Env)。
    
    [任务 1.1] 更新:
    这个环境现在期望一个包含 L2 分析特征 (例如 l2_sentiment, l2_confidence) 
    的 DataFrame，并将它们包含在 observation_space 中。
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, config: Dict[str, Any]):
        """
        初始化环境。
        
        config 必须包含:
        - df (pd.DataFrame): 包含价格数据和 [任务 1.1] L2 特征的 DataFrame。
        - agent_id (str): 智能体的唯一 ID (例如 'alpha_agent', 'risk_agent')。
        - initial_balance (float): 初始现金。
        - action_space_config (Dict):
            - type (str): 'discrete' (例如 [Buy, Sell, Hold]) 或 'continuous' (例如 [-1.0, 1.0])。
            - n (int): 如果是 'discrete'，则为动作数量。
        - (其他特定于 agent_id 的配置...)
        """
        super().__init__()
        
        self.df = config["df"]
        self.agent_id = config.get("agent_id", "default_agent")
        self.initial_balance = config.get("initial_balance", 100000.0)
        
        # [任务 1.1] 扩展 Observation Space
        # 假设 df columns 包含: 'price', 'l2_sentiment', 'l2_confidence'
        # 状态 (5-d): [balance, shares_held, price, l2_sentiment, l2_confidence]
        
        # (我们使用低=-1，高=inf 来处理潜在的负情绪得分和无限的余额)
        self.observation_space = spaces.Box(
            low=-1.0, high=np.inf, shape=(5,), dtype=np.float32
        )

        # (从配置中定义 Action Space)
        action_config = config.get("action_space_config", {"type": "discrete", "n": 3})
        if action_config["type"] == "discrete":
            # 示例: 0=Hold, 1=Buy, 2=Sell
            self.action_space = spaces.Discrete(action_config["n"])
        elif action_config["type"] == "continuous":
            # 示例: -1.0 (全卖) 到 1.0 (全买/分配)
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            raise ValueError(f"不支持的 action_space type: {action_config['type']}")

        self.current_step = 0
        self.max_steps = len(self.df) - 1
        
        self.reset()

    def reset(self, *, seed=None, options=None):
        """重置环境到初始状态。"""
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.shares_held = 0.0
        self.total_value = self.initial_balance
        self.current_step = 0
        
        obs = self._get_state()
        info = self._get_info()
        
        return obs, info

    def _get_state(self) -> np.ndarray:
        """
        [任务 1.1] 构建并返回扩展后的状态数组。
        """
        # (从 self.df (DataFrame) 中提取当前行数据)
        try:
            current_data_row = self.df.iloc[self.current_step]
            price = current_data_row.get("price", 0.0)
        except IndexError:
            # (如果到达末尾，使用最后一步的数据)
            current_data_row = self.df.iloc[-1]
            price = current_data_row.get("price", 0.0)

        # (关键) [任务 1.1] 提取新的 L2 特征
        # (我们假设 'l2_sentiment' 是一个数值, 'l2_confidence' 也是)
        # (如果训练数据中缺少这些列，则提供合理的默认值)
        sentiment = current_data_row.get("l2_sentiment", 0.0) 
        confidence = current_data_row.get("l2_confidence", 0.5) # (中性信心)

        # 状态 (5-d): [balance, shares_held, price, l2_sentiment, l2_confidence]
        state = np.array([
            self.balance,
            self.shares_held,
            price,
            sentiment,
            confidence
        ], dtype=np.float32)
        
        return state

    def _get_info(self) -> Dict[str, Any]:
        """返回关于当前状态的附加信息 (RLLib 可选)。"""
        return {
            "step": self.current_step,
            "total_value": self.total_value,
            "shares_held": self.shares_held,
            "balance": self.balance
        }

    def step(self, action):
        """
        执行一个时间步。
        
        (注意: 这里的交易逻辑 (trade_logic) 需要根据您的
         Alpha/Risk/Exec 智能体的具体动作定义进行实现。)
        """
        
        # (示例: 基于动作执行交易的简化逻辑)
        # (您需要根据您的动作空间定义来实现这个 trade_logic)
        # ... trade_logic(action) ...
        
        # (更新总价值)
        current_price = self.df.iloc[self.current_step].get("price", 0.0)
        self.total_value = self.balance + (self.shares_held * current_price)
        
        # (计算奖励 - 示例: 价值变化)
        reward = self.total_value - (self.balance + self.shares_held * self.df.iloc[self.current_step - 1].get("price", 0.0))
        
        # (进入下一步)
        self.current_step += 1
        
        # (检查是否完成)
        terminated = self.current_step >= self.max_steps
        truncated = False # (如果需要，可以实现截断逻辑)
        
        obs = self._get_state()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Total Value: {self.total_value}")
            print(f"Shares Held: {self.shares_held}")
            print(f"Balance: {self.balance}")

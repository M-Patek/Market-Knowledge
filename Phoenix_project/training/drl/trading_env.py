# (原: drl/trading_env.py)
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

# --- [修复] ---
# 修复：将相对导入 'from ...data_manager...' 更改为绝对导入
from data_manager import DataManager
# 修复：将相对导入 'from ...core.schemas.data_schema...' 更改为绝对导入
from core.schemas.data_schema import TickerData
# 修复：将相对导入 'from ...monitor.logging...' 更改为绝对导入
from monitor.logging import get_logger
# --- [修复结束] ---

logger = get_logger(__name__)

class TradingEnv(gym.Env):
    """
    一个与 Gymnasium (OpenAI Gym) 兼容的交易环境。
    用于 DRL 智能体的离线训练 (回测模拟)。
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, config: Dict[str, Any], data_manager: DataManager):
        super(TradingEnv, self).__init__()
        
        self.config = config
        self.env_config = config.get('drl_env', {})
        self.data_manager = data_manager
        
        self.start_date = config.get('start_date')
        self.end_date = config.get('end_date')
        self.asset_universe = config.get('asset_universe', ['AAPL']) # 简化：单个资产
        self.symbol = self.asset_universe[0]
        
        self.initial_balance = self.env_config.get('initial_balance', 10000.0)
        self.lookback_window = self.env_config.get('lookback_window', 30) # 状态包含过去 30 天
        
        # 动作空间：-1 (卖出), 0 (持有), 1 (买入)
        # 简化：离散动作
        self.action_space = spaces.Discrete(3) # 0: Sell, 1: Hold, 2: Buy

        # 状态空间：(价格数据[lookback_window, features] + 持有量[1] + 余额[1])
        # 简化：仅价格
        self.observation_space = spaces.Box(
            low=0, 
            high=np.inf, 
            shape=(self.lookback_window, 5), # 5 个特征 (O, H, L, C, V)
            dtype=np.float32
        )
        
        self.historical_data = None
        self.current_step = 0
        self.end_step = 0
        
        self._load_data()

    def _load_data(self):
        """
        (同步) 加载 DRL 训练所需的所有历史数据。
        注意：在真实场景中，DataManager 可能是异步的。
        """
        logger.info("DRL 环境：正在加载历史数据...")
        try:
            # 简化：假设 DataManager 有一个同步方法或我们在此处处理异步
            # 在一个真实的 asyncio 应用中，我们可能需要一个事件循环
            # 但 SB3/Gym 通常是同步的。
            
            # 模拟同步加载
            # self.historical_data = self.data_manager.get_sync_data(
            #     self.symbol, self.start_date, self.end_date
            # )
            
            # 模拟数据
            dates = pd.date_range(self.start_date, self.end_date, freq='B')
            data = np.random.randn(len(dates), 5)
            data[:, 0:4] = np.abs(data[:, 0:4] + 100) # OHLC
            data[:, 4] = np.abs(data[:, 4] * 10000) # Volume
            self.historical_data = pd.DataFrame(data, columns=['open', 'high', 'low', 'close', 'volume'], index=dates)
            
            if self.historical_data is None or len(self.historical_data) < self.lookback_window:
                raise ValueError("加载的数据不足。")
            
            self.end_step = len(self.historical_data) - 1
            logger.info(f"数据已加载：{len(self.historical_data)} 条记录。")
            
        except Exception as e:
            logger.error(f"DRL 环境数据加载失败: {e}", exc_info=True)
            raise

    def reset(self, seed: Optional[int] = None):
        super().reset(seed=seed) # 处理种子
        
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.current_step = self.lookback_window # 从第N天开始，才有足够的回溯数据
        
        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def _get_observation(self):
        """
        获取当前步骤的观测状态。
        """
        start = self.current_step - self.lookback_window
        end = self.current_step
        obs_data = self.historical_data.iloc[start:end][['open', 'high', 'low', 'close', 'volume']]
        return obs_data.values.astype(np.float32)

    def _get_info(self):
        """
        获取当前步骤的辅助信息。
        """
        return {
            "step": self.current_step,
            "net_worth": self.net_worth,
            "shares_held": self.shares_held,
            "balance": self.balance
        }

    def step(self, action):
        # action: 0=Sell, 1=Hold, 2=Buy
        
        current_price = self.historical_data['close'].iloc[self.current_step]
        
        # 执行动作
        if action == 0: # Sell
            if self.shares_held > 0:
                self.balance += self.shares_held * current_price
                self.shares_held = 0
        elif action == 2: # Buy
            if self.balance > current_price:
                # 简化：一次买 1 股
                self.shares_held += 1
                self.balance -= current_price
        # action == 1 (Hold) -> 不执行任何操作

        # 计算奖励
        new_net_worth = self.balance + (self.shares_held * current_price)
        reward = new_net_worth - self.net_worth # 奖励 = 净值变化
        self.net_worth = new_net_worth
        
        # 进入下一步
        self.current_step += 1
        
        # 检查是否结束
        terminated = self.net_worth <= 0 or self.current_step >= self.end_step
        truncated = False # 我们不截断

        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info

    def render(self, mode='human'):
        if mode == 'human':
            print(f"Step: {self.current_step}, Net Worth: {self.net_worth:.2f}, "
                  f"Shares: {self.shares_held}, Balance: {self.balance:.2f}")

    def close(self):
        pass

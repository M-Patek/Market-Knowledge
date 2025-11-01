import numpy as np
import pandas as pd
# 修正: 'gym' 已被 'gymnasium' 替代。我们已在 requirements.txt 中添加 gymnasium
import gymnasium as gym
from gymnasium import spaces
from typing import List, Dict, Any, Optional

from data.data_iterator import DataIterator
from execution.order_manager import OrderManager
from core.pipeline_state import PipelineState

class TradingEnv(gym.Env):
    """
    一个符合 OpenAI Gym 规范的交易环境，用于强化学习。
    (Task 1.2 - DRL 基础设施)
    
    这个环境模拟了在一个时间序列上进行买入/卖出/持有的决策过程。
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                 data_iterator: DataIterator, 
                 order_manager: OrderManager,
                 initial_capital: float = 100000.0,
                 lookback_window: int = 30,
                 **kwargs):
        """
        初始化交易环境。

        Args:
            data_iterator: 提供市场数据的迭代器。
            order_manager: 处理订单执行和投资组合管理的对象。
            initial_capital: 模拟开始时的初始资金。
            lookback_window: 代理在每一步可以看到的历史数据天数。
            **kwargs: 传递给父类的其他参数 (例如 'ticker')。
        """
        super(TradingEnv, self).__init__()
        
        self.data_iterator = data_iterator
        self.order_manager = order_manager
        self.initial_capital = initial_capital
        self.lookback_window = lookback_window
        
        # 从迭代器获取特征数量
        # (lookback_window, num_features)
        self.num_features = self.data_iterator.get_feature_count()
        
        # 动作空间: 0: 卖出, 1: 持有, 2: 买入
        self.action_space = spaces.Discrete(3) 
        
        # 观测空间: (窗口大小, 特征数量)
        # 例如: 过去30天的 (Open, High, Low, Close, Volume, RSI, SMA)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.lookback_window, self.num_features), 
            dtype=np.float32
        )
        
        # 内部状态
        self.current_step = 0
        self.done = False
        self.portfolio_value = self.initial_capital
        self.return_history = [] # 存储每日回报

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[np.ndarray, dict]:
        """
        重置环境到初始状态。
        """
        if seed is not None:
            super().reset(seed=seed)
            
        self.current_step = 0
        self.done = False
        self.data_iterator.reset()
        self.order_manager.reset_portfolio(self.initial_capital)
        self.portfolio_value = self.initial_capital
        self.return_history = []

        # 获取初始观测
        obs = self._get_observation()
        
        # 填充回看窗口 (在模拟开始前)
        # 我们跳过前 `lookback_window` 步来预热状态
        for _ in range(self.lookback_window):
             _, done = self.data_iterator.next()
             if done:
                 raise ValueError("数据集太短，无法填充初始回看窗口")
        
        obs = self._get_observation()
        self.current_step = self.lookback_window
        
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        在环境中执行一个步骤 (一个交易日)。

        Args:
            action (int): 代理选择的动作 (0=卖, 1=持有, 2=买)。

        Returns:
            tuple: (observation, reward, done, truncated, info)
        """
        if self.done:
            # 如果环境已结束，返回最后的状态
            obs = self._get_observation()
            return obs, 0.0, self.done, False, self._get_info()

        # 1. 获取当前市场数据 (用于执行)
        current_data_slice, self.done = self.data_iterator.next()
        if self.done:
            # 数据结束
            obs = self._get_observation()
            return obs, 0.0, self.done, False, self._get_info()
            
        self.current_step += 1
        
        # 2. 获取当前投资组合价值 (T-1)
        value_t_minus_1 = self.order_manager.get_portfolio_value(current_data_slice)

        # 3. 将 DRL 动作转换为 OrderManager 信号
        # 这是一个简化的映射：
        # 0 (卖): 平仓所有多头头寸
        # 1 (持有): 不做操作
        # 2 (买): 分配 100% 资金做多
        
        signal_pct = 0.0 # 默认为持有 (或平仓)
        if action == 2: # 买入
            signal_pct = 1.0
        elif action == 0: # 卖出
            signal_pct = 0.0 # 目标仓位为 0%
            
        # 假设我们只交易 data_iterator.ticker
        ticker = self.data_iterator.get_current_ticker()
        if ticker:
            # 生成目标信号
            target_signal = {ticker: signal_pct}
            
            # 4. OrderManager 执行交易
            # OrderManager 会处理滑点、佣金和投资组合更新
            self.order_manager.process_signals(
                signals=target_signal, 
                data_slice=current_data_slice
            )

        # 5. 获取新的投资组合价值 (T)
        self.portfolio_value = self.order_manager.get_portfolio_value(current_data_slice)
        
        # 6. 计算奖励 (Reward)
        # 奖励 = (T 时价值) - (T-1 时价值)
        reward = self.portfolio_value - value_t_minus_1
        self.return_history.append(reward) # 存储绝对回报

        # 7. 获取下一个观测
        obs = self._get_observation()
        
        # 8. 获取信息
        info = self._get_info()

        return obs, reward, self.done, False, info

    def _get_observation(self) -> np.ndarray:
        """
        从数据迭代器获取当前的回看窗口。
        """
        return self.data_iterator.get_lookback_window(self.lookback_window)

    def _get_info(self) -> dict:
        """
        返回关于当前步骤的附加信息。
        """
        return {
            "step": self.current_step,
            "portfolio_value": self.portfolio_value,
            "cash": self.order_manager.portfolio.get('CASH', 0),
            "positions": self.order_manager.portfolio
        }

    def render(self, mode='human'):
        """
        (可选) 渲染环境状态。
        """
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: {self.portfolio_value:.2f}")
            print(f"Positions: {self.order_manager.portfolio}")

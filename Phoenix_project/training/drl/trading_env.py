import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from gymnasium import spaces

# [MARL 重构] 导入 PettingZoo ParallelEnv
from pettingzoo.utils.env import ParallelEnv

# 修复 (第 4 阶段): 将 TickerData 重命名为 MarketData
from Phoenix_project.core.schemas.data_schema import MarketData

class TradingEnv(ParallelEnv):
    """
    [MARL 重构]
    基于 PettingZoo ParallelEnv 的多智能体股票交易环境。

    这个环境模拟三个智能体（Alpha, Risk, Execution）的协作。
    所有智能体同时行动，环境按顺序处理它们的动作。

    智能体 (Agents):
    - "alpha_agent": 决定 (0=卖, 1=持有, 2=买)
    - "risk_agent": 决定 (0=批准, 1=否决)
    - "execution_agent": 决定 (0=市价单, 1=限价单, 2=拆分)

    状态 (State) - (对所有智能体共享):
    - [账户余额, 持有股数, 当前价格]

    奖励 (Reward) - (共享):
    - 投资组合净值的变化
    """
    metadata = {'render_modes': ['human', 'array'], "name": "phoenix_marl_v0"}

    def __init__(self,
                 data: Dict[str, pd.DataFrame],
                 initial_balance: float = 10000.0,
                 symbol: str = 'DEFAULT',
                 transaction_cost_pct: float = 0.001):
        """
        初始化环境。
        """
        super().__init__()

        self.initial_balance = initial_balance
        self.symbol = symbol
        self.transaction_cost_pct = transaction_cost_pct
        
        # 准备数据
        self.df = data.get(symbol)
        if self.df is None:
            raise ValueError(f"Data for symbol '{symbol}' not found.")
            
        # 确保 timestamp 是 datetime 对象 (如果它来自 env_config)
        if not pd.api.types.is_datetime64_any_dtype(self.df.index):
             if 'timestamp' in self.df.columns:
                 self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
                 self.df = self.df.sort_values('timestamp').set_index('timestamp')
             else:
                raise ValueError("Data must have a 'timestamp' column or index.")

        
        # 将 DataFrame 转换为 MarketData 对象列表 (模拟数据迭代器)
        self.market_data_stream: List[MarketData] = self._df_to_market_data(self.df)
        
        self.total_steps = len(self.market_data_stream) - 1

        # [MARL 重构] 定义智能体
        self.agents = ["alpha_agent", "risk_agent", "execution_agent"]
        self.possible_agents = self.agents[:]

        # [MARL 重构] 定义状态空间
        # 共享相同的状态空间 [余额, 持股, 当前价格]
        low_state = np.array([0, 0, 0], dtype=np.float32)
        high_state = np.array([initial_balance * 10, 1e6, 1e6], dtype=np.float32)
        obs_space = spaces.Box(low=low_state, high=high_state, dtype=np.float32)
        
        # PettingZoo API 要求 observation_spaces 是一个字典
        self.observation_spaces = {agent: obs_space for agent in self.agents}

        # [MARL 重构] 定义动作空间 (根据指令)
        self.action_spaces = {
            "alpha_agent": spaces.Discrete(3), # 0: 卖出, 1: 持有, 2: 买入
            "risk_agent": spaces.Discrete(2),   # 0: 批准, 1: 否决
            "execution_agent": spaces.Discrete(3) # 0: 市价, 1: 限价, 2: 拆分 (简化模拟)
        }

        # 初始化状态 (在 reset 中完成)
        self.balance = 0.0
        self.shares_held = 0
        self.net_worth = 0.0
        self.current_step = 0

    # [MARL 重构] PettingZoo API 方法
    def observation_space(self, agent: str) -> spaces.Space:
        return self.observation_spaces[agent]

    # [MARL 重构] PettingZoo API 方法
    def action_space(self, agent: str) -> spaces.Space:
        return self.action_spaces[agent]

    def _df_to_market_data(self, df: pd.DataFrame) -> List[MarketData]:
        """(辅助函数) 将 DataFrame 转换为 MarketData 列表。"""
        data_list = []
        for timestamp, row in df.iterrows():
            md = MarketData(
                symbol=self.symbol,
                timestamp=timestamp,
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume']
            )
            data_list.append(md)
        return data_list

    def _get_current_data(self) -> Optional[MarketData]:
        """获取当前步骤的数据。"""
        if self.current_step < self.total_steps:
            return self.market_data_stream[self.current_step]
        return None

    def _get_current_price(self) -> float:
        """获取当前价格 (使用 'close')。"""
        data = self._get_current_data()
        if data:
            return data.close
        # 如果数据流结束，使用最后已知的价格
        if self.market_data_stream:
            return self.market_data_stream[-1].close
        return 0.0

    def _get_state(self) -> np.ndarray:
        """构建并返回当前状态。"""
        state = np.array([
            self.balance,
            self.shares_held,
            self._get_current_price()
        ], dtype=np.float32)
        return state

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        [MARL 重构]
        重置环境。
        返回: (observations, infos) 字典
        """
        if seed is not None:
            super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.current_step = 0
        
        # PettingZoo API 要求返回字典
        observations = {agent: self._get_state() for agent in self.agents}
        infos = {agent: {"net_worth": self.net_worth} for agent in self.agents}
        
        return observations, infos

    def step(self, actions: Dict[str, int]) -> Tuple[
        Dict[str, np.ndarray], 
        Dict[str, float], 
        Dict[str, bool], 
        Dict[str, bool], 
        Dict[str, Any]
    ]:
        """
        [MARL 重构]
        执行一个时间步。
        输入: actions 是一个字典: {"alpha_agent": 1, "risk_agent": 0, ...}
        返回: (obs, rewards, terminated, truncated, infos) 字典
        """
        
        if self.current_step >= self.total_steps:
            # 如果环境已结束但仍被调用 step
            observations = {agent: self._get_state() for agent in self.agents}
            rewards = {agent: 0.0 for agent in self.agents}
            terminations = {agent: True for agent in self.agents}
            truncations = {agent: False for agent in self.agents}
            infos = {agent: {} for agent in self.agents}
            return observations, rewards, terminations, truncations, infos

        current_price = self._get_current_price()
        prev_net_worth = self.net_worth

        # 1. 解码智能体动作
        alpha_action = actions.get("alpha_agent", 1) # 0=卖, 1=持有, 2=买
        risk_action = actions.get("risk_agent", 0)   # 0=批准, 1=否决
        exec_action = actions.get("execution_agent", 0) # 0=市价, 1=限价, 2=拆分

        # 2. [MARL 交互逻辑]
        final_action = alpha_action
        
        # 风险智能体否决
        if risk_action == 1: # 否决
            final_action = 1 # 强制 "持有"

        # 3. [MARL 执行逻辑]
        # 根据执行策略模拟不同的成本
        slippage = 0.0
        if final_action != 1: # 如果不是 "持有"
            if exec_action == 0: # 市价单 = 高滑点
                slippage = 0.002
            elif exec_action == 1: # 限价单 = 低滑点 (假设成交)
                slippage = 0.0005
            elif exec_action == 2: # 拆分 = 中滑点
                slippage = 0.001
        
        # 4. 应用最终动作
        if final_action == 0: # 卖出
            if self.shares_held > 0:
                # 简化：卖出 1 股
                self.balance += current_price * (1 - self.transaction_cost_pct - slippage)
                self.shares_held -= 1
        
        elif final_action == 2: # 买入
            if self.balance >= current_price:
                # 简化：买入 1 股
                self.balance -= current_price * (1 + self.transaction_cost_pct + slippage)
                self.shares_held += 1

        # 5. 计算奖励 (共享奖励)
        self.net_worth = self.balance + (self.shares_held * current_price)
        reward = self.net_worth - prev_net_worth
        
        # 6. 移动到下一步
        self.current_step += 1
        
        # 7. 准备 PettingZoo API 的返回
        terminated = (self.net_worth <= 0) or (self.current_step >= self.total_steps)
        truncated = False # 破产被视为 terminated
        
        observations = {agent: self._get_state() for agent in self.agents}
        rewards = {agent: reward for agent in self.agents}
        terminations = {agent: terminated for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        infos = {agent: {"net_worth": self.net_worth} for agent in self.agents}
        
        return observations, rewards, terminations, truncations, infos

    def render(self, mode='human'):
        """渲染环境 (可选)。"""
        if mode == 'human':
            price = self._get_current_price()
            print(f"--- Step: {self.current_step} ---")
            print(f"  Price: {price:.2f}")
            print(f"  Balance: {self.balance:.2f}")
            print(f"  Shares Held: {self.shares_held}")
            print(f"  Net Worth: {self.net_worth:.2f}")

    def close(self):
        """关闭环境。"""
        pass

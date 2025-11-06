import gymnasium as gym
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any
from gymnasium import spaces

# 修复 (第 4 阶段): 将 TickerData 重命名为 MarketData
from Phoenix_project.core.schemas.data_schema import MarketData

# 假设:
# 1. 'MarketData' 具有与 'TickerData' 相同的属性 
#    (symbol, timestamp, open, high, low, close, volume)
# 2. 我们有一个函数可以获取下一个 MarketData, 类似于:
#    self.data_iterator.get_next_market_data(symbol) -> Optional[MarketData]

class TradingEnv(gym.Env):
    """
    基于 Gymnasium 的股票交易环境 (单代理)。

    这个环境模拟单个股票的交易。
    代理 (Agent) 决定是买入、卖出还是持有。

    状态 (State):
    - [账户余额, 持有股数, 当前价格]

    动作 (Action):
    - 0: 卖出 (1 单位)
    - 1: 持有
    - 2: 买入 (1 单位)
    """
    metadata = {'render_modes': ['human', 'array']}

    def __init__(self,
                 data: Dict[str, pd.DataFrame],
                 initial_balance: float = 10000.0,
                 symbol: str = 'DEFAULT',
                 transaction_cost_pct: float = 0.001):
        """
        初始化环境。

        Args:
            data (Dict[str, pd.DataFrame]): 预加载的市场数据字典，键为股票代码。
            initial_balance (float): 初始账户余额。
            symbol (str): 此环境实例交易的股票代码。
            transaction_cost_pct (float): 交易成本百分比。
        """
        super(TradingEnv, self).__init__()

        self.initial_balance = initial_balance
        self.symbol = symbol
        self.transaction_cost_pct = transaction_cost_pct
        
        # 准备数据
        self.df = data.get(symbol)
        if self.df is None:
            raise ValueError(f"Data for symbol '{symbol}' not found.")
            
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        self.df = self.df.sort_values('timestamp').set_index('timestamp')
        
        # 将 DataFrame 转换为 MarketData 对象列表 (模拟数据迭代器)
        self.market_data_stream: List[MarketData] = self._df_to_market_data(self.df)
        
        self.current_step = 0
        self.total_steps = len(self.market_data_stream) - 1 # 减 1 因为我们需要看下一个价格

        # 定义动作空间 (卖出, 持有, 买入)
        self.action_space = spaces.Discrete(3)

        # 定义状态空间 [余额, 持股, 当前价格]
        # 余额和持股可以很大，价格也是
        # 我们使用 Box 空间，设置合理的上界
        low_state = np.array([0, 0, 0], dtype=np.float32)
        high_state = np.array([initial_balance * 10, 1e6, 1e6], dtype=np.float32) # 假设
        self.observation_space = spaces.Box(low=low_state, high=high_state, dtype=np.float32)

        # 初始化状态
        self.reset()

    def _df_to_market_data(self, df: pd.DataFrame) -> List[MarketData]:
        """(辅助函数) 将 DataFrame 转换为 MarketData 列表。"""
        data_list = []
        for timestamp, row in df.iterrows():
            # 修复 (第 4 阶段): 实例化 MarketData
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
            # 修复 (第 4 阶段): 从流中获取 MarketData
            return self.market_data_stream[self.current_step]
        return None

    def _get_current_price(self) -> float:
        """获取当前价格 (使用 'close')。"""
        data = self._get_current_data()
        if data:
            return data.close
        return 0.0 # 或者处理错误

    def _get_state(self) -> np.ndarray:
        """构建并返回当前状态。"""
        state = np.array([
            self.balance,
            self.shares_held,
            self._get_current_price()
        ], dtype=np.float32)
        return state

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """重置环境到初始状态。"""
        super().reset(seed=seed)
        
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.current_step = 0
        
        # 返回初始状态和信息
        info = {}
        return self._get_state(), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        执行一个时间步。

        Returns:
            Tuple[np.ndarray, float, bool, bool, dict]: 
            (state, reward, terminated, truncated, info)
        """
        terminated = False
        truncated = False
        reward = 0.0
        
        if self.current_step >= self.total_steps:
            # 数据结束
            terminated = True
            return self._get_state(), 0.0, terminated, truncated, {}

        current_price = self._get_current_price()
        
        # 1. 执行动作
        if action == 0: # 卖出
            if self.shares_held > 0:
                # 卖出 1 股
                self.balance += current_price * (1 - self.transaction_cost_pct)
                self.shares_held -= 1
        
        elif action == 2: # 买入
            if self.balance >= current_price:
                # 买入 1 股
                self.balance -= current_price * (1 + self.transaction_cost_pct)
                self.shares_held += 1

        # (action == 1: 持有 - 无操作)

        # 2. 计算净值 (用于奖励)
        prev_net_worth = self.net_worth
        self.net_worth = self.balance + (self.shares_held * current_price)
        
        # 3. 计算奖励 (基于净值的变化)
        reward = self.net_worth - prev_net_worth
        
        # 4. 移动到下一步
        self.current_step += 1
        
        # 5. 检查是否终止
        if self.current_step >= self.total_steps:
            terminated = True
        
        # (截断逻辑, e.g., if self.balance <= 0)
        if self.balance <= 0:
            truncated = True # 破产

        info = {'net_worth': self.net_worth}
        
        return self._get_state(), reward, terminated, truncated, info

    def render(self, mode='human'):
        """渲染环境 (可选)。"""
        if mode == 'human':
            price = self._get_current_price()
            print(f"--- Step: {self.current_step} ---")
            print(f"  Price: {price:.2f}")
            print(f"  Balance: {self.balance:.2f}")
            print(f"  Shares Held: {self.shares_held}")
            print(f"  Net Worth: {self.net_worth:.2f}")

# (原: drl/execution_env.py)
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# --- [修复] ---
# 原: from ..execution.interfaces import Order, Fill
# 新: from ...execution.interfaces import Order, Fill (training/drl/ -> ... -> execution/)
# --- [修复结束] ---
from ...execution.interfaces import Order, Fill
from ...monitor.logging import get_logger

logger = get_logger(__name__)

class ExecutionEnv(gym.Env):
    """
    一个用于训练“执行智能体” (ExecutionAgent) 的 Gym 环境。
    
    目标：在给定一个大的目标订单（例如“买入 1000 股 AAPL”）后，
    智能体必须学会在一段时间内（例如 1 小时）将其拆分成小订单，
    以最小化市场冲击（Slippage）。
    """
    
    def __init__(self, config: Dict[str, Any]):
        super(ExecutionEnv, self).__init__()
        
        self.config = config.get('exec_env', {})
        
        # 动作空间：(拆分比例, 价格限制)
        # 简化：在 5 个时间步内执行完毕，每步执行 20%
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32) # 每一步执行的百分比
        
        # 状态空间：(剩余订单量, 剩余时间, LOB快照)
        # 简化：(剩余订单百分比, 剩余时间步)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)

        self.total_steps = self.config.get('total_steps', 5) # 5 步内执行完
        
        logger.info("ExecutionEnv (订单执行环境) 已初始化。")

    def reset(self, seed: Optional[int] = None):
        super().reset(seed=seed)
        
        self.current_step = 0
        self.remaining_pct = 1.0 # 剩余 100% 的订单
        
        # 模拟一个 LOB (Level 2 Order Book)
        self.lob = self._generate_lob()
        
        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def _get_observation(self):
        return np.array([
            self.remaining_pct,
            (self.total_steps - self.current_step) / self.total_steps
        ]).astype(np.float32)
        
    def _get_info(self):
        return {"step": self.current_step, "remaining_pct": self.remaining_pct}

    def _generate_lob(self):
        """模拟生成一个 LOB 快照"""
        return {"bids": [(100, 50), (99, 100)], "asks": [(101, 70), (102, 80)]}
        
    def _calculate_slippage(self, trade_pct: float):
        """模拟市场冲击"""
        # 冲击是交易量的平方
        return (trade_pct * 100) ** 2 * 0.001

    def step(self, action):
        # action[0] 是智能体想要在这一步执行的百分比
        trade_pct = np.clip(action[0], 0, self.remaining_pct)
        
        self.remaining_pct -= trade_pct
        self.current_step += 1
        
        # 计算奖励 (目标：最小化冲击)
        # 冲击 (Slippage) 是负奖励
        slippage_cost = self._calculate_slippage(trade_pct)
        reward = -slippage_cost
        
        terminated = self.current_step >= self.total_steps
        
        if terminated and self.remaining_pct > 0.01:
            # 惩罚：未能在规定时间内执行完
            reward -= self.remaining_pct * 100 # 巨大惩罚
            
        truncated = False
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Remaining: {self.remaining_pct:.2%}")

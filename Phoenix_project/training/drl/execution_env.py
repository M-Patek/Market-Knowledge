import numpy as np
# 修正: 'gym' 已被 'gymnasium' 替代。我们已在 requirements.txt 中添加 gymnasium
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, Tuple, Optional

class ExecutionEnv(gym.Env):
    """
    一个符合 OpenAI Gym 规范的订单执行环境。
    (Task 1.2 - DRL 基础设施)
    
    这个环境的目标是为 'ExecutionAgent' 训练一个模型，
    使其学会如何将一个大的父订单（例如 "买入 10000 股 AAPL"）
    拆分成多个子订单，以在给定的时间窗口内（例如 30 分钟）
    最小化市场影响（滑点）。
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, 
                 parent_order_size: float = 10000.0,
                 time_steps: int = 30, # 例如, 30 个 1 分钟的时间步
                 market_impact_coeff: float = 0.001, # 市场影响系数
                 **kwargs):
        """
        初始化执行环境。

        Args:
            parent_order_size: 需要执行的总订单大小 (例如 10000 股)。
            time_steps: 将父订单拆分成的子订单数量 (例如 30 步)。
            market_impact_coeff: 用于模拟滑点的系数。
        """
        super(ExecutionEnv, self).__init__()
        
        self.start_order_size = parent_order_size
        self.total_time_steps = time_steps
        self.market_impact_coeff = market_impact_coeff

        # 动作空间: 连续空间，[0, 1]
        # 代理决定在当前时间步执行 *剩余订单* 的百分比。
        # 0 = 不执行, 1 = 执行所有剩余订单
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # 观测空间:
        # [
        #   0: 剩余未执行的订单比例 (1.0 -> 0.0),
        #   1: 剩余时间步比例 (1.0 -> 0.0)
        # ]
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(2,), dtype=np.float32
        )
        
        # 内部状态
        self.remaining_order_size = self.start_order_size
        self.current_step = 0
        self.base_price = 100.0 # 假设一个稳定的基础价格

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        重置环境到初始状态。
        """
        if seed is not None:
            super().reset(seed=seed)
            
        self.remaining_order_size = self.start_order_size
        self.current_step = 0
        
        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        在环境中执行一个步骤 (一个执行切片)。

        Args:
            action (np.ndarray): 代理选择的动作 (执行剩余订单的百分比, [0, 1])。

        Returns:
            tuple: (observation, reward, done, truncated, info)
        """
        
        # 1. 确定要执行的订单大小
        execution_percentage = action[0]
        # 确保百分比在 [0, 1] 之间
        execution_percentage = np.clip(execution_percentage, 0, 1)
        
        # 如果这是最后一步，强制执行所有剩余订单
        if self.current_step == self.total_time_steps - 1:
            execution_percentage = 1.0
            
        order_to_execute = self.remaining_order_size * execution_percentage
        
        # 2. 计算市场影响 (滑点)
        # 滑点 = 基础价格 * (执行订单大小 / 总订单大小)^0.5 * 市场影响系数
        # 这是一个简化的 Almgren-Chriss 模型
        slippage_per_share = self.base_price * \
                             self.market_impact_coeff * \
                             np.sqrt(order_to_execute / self.start_order_size)
        
        # 3. 计算奖励 (Reward)
        # 奖励是负的，等于总滑点成本。目标是最大化这个（负）奖励（即最小化成本）。
        # 奖励 = -(执行的股数 * 每股滑点)
        reward = -1 * (order_to_execute * slippage_per_share)
        
        # 4. 更新内部状态
        self.remaining_order_size -= order_to_execute
        self.current_step += 1
        
        # 5. 检查是否完成
        done = self.current_step >= self.total_time_steps
        
        # 6. 获取下一个观测和信息
        obs = self._get_observation()
        info = self._get_info()
        info["slippage_cost"] = -reward # 记录正的成本

        return obs, reward, done, False, info

    def _get_observation(self) -> np.ndarray:
        """
        获取当前状态的观测。
        """
        remaining_order_pct = self.remaining_order_size / self.start_order_size
        remaining_time_pct = (self.total_time_steps - self.current_step) / self.total_time_steps
        
        return np.array([remaining_order_pct, remaining_time_pct], dtype=np.float32)

    def _get_info(self) -> dict:
        """
        返回关于当前步骤的附加信息。
        """
        return {
            "step": self.current_step,
            "remaining_order_size": self.remaining_order_size
        }

    def render(self, mode='human'):
        """
        (可选) 渲染环境状态。
        """
        if mode == 'human':
            print(f"Step: {self.current_step}/{self.total_time_steps}")
            print(f"Remaining Order: {self.remaining_order_size:.2f} / {self.start_order_size:.2f}")

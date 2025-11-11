# training/drl/phoenix_env_v7.py
# [凤凰 V7 迭代] - 统一效用与务实权衡
# 最终的、无幻觉的 DRL 环境。
# 替代所有 V1-V6 的幻觉设计。

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any

# --- V7 奖励函数惩罚系数 (可调参数) ---

# R_Risk_Component (风险组件)
# K_DRAWDOWN: 对回撤的平方惩罚系数。RenTec 铁律。
K_DRAWDOWN = 10.0
# K_UNCERTAINTY: 对“L2 不确定性 * 仓位规模”的平方惩罚系数。
# 这是 V7 的“前瞻性风险”核心。
K_UNCERTAINTY = 5.0 

# --- V7 凤凰多智能体环境 ---

class PhoenixMultiAgentEnvV7(gym.Env):
    """
    凤凰 V7 多智能体 DRL 环境 (PhoenixMultiAgentEnvV7)
    
    该环境实现了“统一效用与务实权衡” (V7) 奖励哲学。
    
    - 统一效用: 所有 L3 智能体 (Alpha, Risk, Exec) 共享
      同一个奖励信号 `Total_Reward_V7`。
    - 即时反馈 (Dense): 奖励在每一步 (t+1) 立即计算。
    - 无幻觉:
        1. 不依赖“未来”PnL (V3 幻觉)。
        2. 不依赖“全知 L2”计算的“目标投资组合” (V6 幻觉)。
        3. 不依赖“缓慢”的 EMA 信号 (V5 幻觉)。
        
    - V7 奖励 = (知识调整后 PnL) - (成本) - (风险惩罚)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化 V7 环境。
        
        Config 必须包含:
        - data_iterator (Object): 迭代器，返回 `(timestamp, data_batch)`。
        - orchestrator (Object): L1/L2 编排器 (用于获取 L2 知识)。
        - context_bus (Object): 系统的上下文总线 (用于获取投资组合状态)。
        - initial_balance (float): 初始现金。
        """
        super().__init__()
        
        # 1. 核心组件 (从 phoenix_project.py 传入)
        self.data_iterator = config["data_iterator"]
        self.orchestrator = config["orchestrator"]
        self.context_bus = config["context_bus"]
        
        # 2. 智能体定义
        # (V7 哲学：它们是单一团队，但 RLLib 仍将它们视为多智能体)
        self.agents = ["alpha", "risk", "exec"]
        
        # 3. 定义 Observation 和 Action Spaces (示例)
        # (这需要根据您的 L3 智能体的实际输入/输出进行详细定义)
        # (这里使用简化的占位符)
        self._obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        # 示例：动作空间 {asset_id: target_allocation}
        self._action_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32) # 假设 5 种资产

        self.observation_space = spaces.Dict({agent: self._obs_space for agent in self.agents})
        self.action_space = spaces.Dict({agent: self._action_space for agent in self.agents})

        # 4. V7 奖励函数状态变量
        self.initial_balance = config.get("initial_balance", 100000.0)
        
        self.reset()

    def reset(self, *, seed=None, options=None):
        """重置环境状态"""
        super().reset(seed=seed)
        
        # (TBD: 重置 data_iterator)
        # self.data_iterator.reset() 
        
        self.balance = self.initial_balance
        self.total_value = self.initial_balance
        
        # V7 风险状态
        self.max_total_value_seen = self.initial_balance
        self.current_drawdown = 0.0
        
        # V7 仓位状态
        # (仓位字典，例如 { 'AAPL': { 'shares': 10, 'avg_price': 150.0 } })
        self.positions = {}
        # (当前投资组合分配向量，例如 { 'AAPL': 0.1, 'MSFT': -0.05 })
        self.current_allocation = {}
        
        self.current_step = 0
        
        # (TBD: 获取初始 L1/L2 状态)
        self.l2_knowledge = {} # 占位符
        
        obs = self._get_obs_dict()
        info = self._get_info_dict()
        
        return obs, info

    def step(self, action_dict: Dict[str, Any]):
        """
        执行一个时间步，并计算 V7 统一奖励。
        """
        self.current_step += 1
        
        # 1. (TBD) 获取 L1/L2 知识 (t 时刻)
        # (在 V7 架构中，L1/L2 必须在 L3 之前运行)
        # try:
        #     timestamp, data_batch = next(self.data_iterator)
        #     self.l2_knowledge = self._run_l1_l2(data_batch)
        # except StopIteration:
        #     return self._terminate_episode()
        
        # (使用占位符数据进行模拟)
        self.l2_knowledge = {
            'AAPL': {'price_t': 150.0, 'price_t1': 150.1, 'l2_sentiment': 0.6, 'l2_confidence': 0.8},
            'MSFT': {'price_t': 300.0, 'price_t1': 299.8, 'l2_sentiment': -0.3, 'l2_confidence': 0.5}
        }

        # 2. (TBD) 执行交易 (t -> t+1)
        # V7 哲学：L3 团队 (Alpha, Risk, Exec) 共同决定一个动作
        # (这里简化为只使用 'alpha' 智能体的动作作为团队决策)
        team_action = action_dict.get('alpha') 
        
        # _execute_trades 必须更新 self.positions, self.balance, self.current_allocation
        # 并且必须返回 t+1 时刻的实际成本
        costs_t1 = self._execute_trades(team_action, self.l2_knowledge)

        # 3. 计算投资组合指标 (t+1)
        old_value = self.total_value
        self.total_value = self._update_portfolio_value(self.l2_knowledge) # 使用 t+1 的价格

        # --- 4. 计算 V7 统一奖励 (t+1 时刻) ---
        
        # 4.A. Alpha & Cost 组件
        R_alpha_cost = self._calculate_alpha_cost_reward(old_value, self.total_value, costs_t1)
        
        # 4.B. 风险组件
        R_risk = self._calculate_risk_reward()
        
        # 4.C. 最终 V7 统一奖励
        Total_Reward_V7 = R_alpha_cost + R_risk

        # 5. 统一分配奖励 (V7 哲学)
        rewards = {agent: Total_Reward_V7 for agent in self.agents}

        # 6. 返回结果
        obs = self._get_obs_dict()
        info = self._get_info_dict(R_alpha_cost, R_risk, Total_Reward_V7)
        
        # (TBD: 定义终止/截断条件)
        terminated = self.total_value <= (self.initial_balance * 0.5) # 50% 回撤
        truncated = self.current_step >= 1000 # 示例
        
        terminateds = {agent: terminated for agent in self.agents}
        terminateds["__all__"] = terminated
        truncateds = {agent: truncated for agent in self.agents}
        truncateds["__all__"] = truncated
        
        return obs, rewards, terminateds, truncateds, info

    # --- V7 奖励函数辅助方法 ---

    def _calculate_alpha_cost_reward(self, old_value: float, new_value: float, costs: Dict[str, float]) -> float:
        """(V7 Alpha+Cost) 计算知识调整后的 PnL 和成本。"""
        
        # 1. (无幻觉) 计算*实际*的 Delta PnL
        # (这是市值（Mark-to-Market）PnL，*不包括*成本)
        delta_pnl_t1 = new_value - old_value
        
        # 2. (无幻觉) 计算*实际*的成本惩罚
        R_cost_penalty_t1 = - (costs.get('slippage', 0.0) + costs.get('fees', 0.0))

        # 3. (V7 核心) 计算“知识调整因子” (L2 Factor)
        # L2_Factor = (1.0 + (l2_sentiment * l2_confidence * sign(Position)))
        # 我们必须计算投资组合的*加权平均*知识因子
        
        l2_factor_total = 0.0
        total_abs_allocation = 0.0001 # 避免除以零
        
        for asset_id, alloc in self.current_allocation.items():
            l2_info = self.l2_knowledge.get(asset_id)
            if not l2_info:
                continue
            
            alloc_abs = abs(alloc)
            total_abs_allocation += alloc_abs
            
            pos_sign = np.sign(alloc)
            l2_sent = l2_info['l2_sentiment']
            l2_conf = l2_info['l2_confidence']
            
            # (1.0 + (sent * conf * sign))
            factor = 1.0 + (l2_sent * l2_conf * pos_sign)
            
            l2_factor_total += (factor * alloc_abs)

        # 加权平均 L2 因子
        avg_l2_factor_t = l2_factor_total / total_abs_allocation

        # 4. 组合 Alpha 组件
        # (如果 PnL 为正，且符合知识，则放大；如果 PnL 为负，且符合知识，则惩罚放大)
        # (如果 PnL 为正，但不符合知识，则抑制)
        R_alpha = delta_pnl_t1 * avg_l2_factor_t
        
        return R_alpha + R_cost_penalty_t1

    def _calculate_risk_reward(self) -> float:
        """(V7 风险) 计算回撤和不确定性惩罚。"""
        
        # 1. (无幻觉) 已实现风险：回撤
        self.max_total_value_seen = max(self.max_total_value_seen, self.total_value)
        self.current_drawdown = (self.max_total_value_seen - self.total_value) / self.max_total_value_seen
        
        R_drawdown_penalty = -K_DRAWDOWN * (self.current_drawdown ** 2)
        
        # 2. (无幻觉) 前瞻性风险：L2 不确定性 * 仓位规模
        
        total_uncertainty_penalty = 0.0
        
        for asset_id, alloc in self.current_allocation.items():
            l2_info = self.l2_knowledge.get(asset_id)
            if not l2_info:
                continue
            
            l2_uncertainty = 1.0 - l2_info['l2_confidence']
            pos_size = abs(alloc) # 分配比例
            
            # V7 核心惩罚：(仓位 * 不确定性)^2
            penalty = (pos_size * l2_uncertainty) ** 2
            total_uncertainty_penalty += penalty
            
        R_uncertainty_penalty = -K_UNCERTAINTY * total_uncertainty_penalty
        
        return R_drawdown_penalty + R_uncertainty_penalty

    # --- 环境辅助方法 (TBD) ---

    def _execute_trades(self, target_allocation_vector: Any, l2_knowledge: Dict) -> Dict[str, float]:
        """(TBD) 模拟交易执行 (t -> t+1)。"""
        # (这是一个复杂的实现)
        # 1. 计算从 self.current_allocation 到 target_allocation_vector 所需的交易
        # 2. 基于 l2_knowledge['price_t'] 计算名义价值
        # 3. 模拟滑点和费用
        # 4. 更新 self.positions (shares, avg_price)
        # 5. 更新 self.balance
        # 6. 更新 self.current_allocation (基于 t+1 的新价值)
        
        # 占位符：模拟更新 allocation
        self.current_allocation = {'AAPL': 0.1, 'MSFT': -0.05}
        
        return {'slippage': 0.01, 'fees': 1.0} # (模拟 0.01 滑点, 1.0 费用)

    def _update_portfolio_value(self, l2_knowledge: Dict) -> float:
        """(TBD) 使用 t+1 的价格计算当前总价值。"""
        value = self.balance
        
        # (这个逻辑需要基于 self.positions 和 t+1 的价格)
        # for asset_id, pos in self.positions.items():
        #     current_price_t1 = l2_knowledge.get(asset_id, {}).get('price_t1')
        #     if current_price_t1:
        #         value += pos['shares'] * current_price_t1
        
        # 占位符：
        if self.current_step > 1:
            value = self.total_value + np.random.randn() * 100 # 模拟价值变化
        
        return value

    def _get_obs_dict(self) -> Dict[str, Any]:
        """(TBD) 为每个智能体构建观察空间。"""
        # 必须包含 L2 知识 (sentiment, confidence)
        # 必须包含当前仓位 (self.current_allocation)
        # 必须包含风险状态 (self.current_drawdown)
        obs = np.random.rand(10).astype(np.float32) # 占位符
        return {agent: obs for agent in self.agents}

    def _get_info_dict(self, r_alpha_cost=0, r_risk=0, r_total=0) -> Dict[str, Any]:
        """(TBD) 返回调试信息。"""
        return {
            "step": self.current_step,
            "total_value": self.total_value,
            "drawdown": self.current_drawdown,
            "reward_total": r_total,
            "reward_alpha_cost": r_alpha_cost,
            "reward_risk": r_risk
        }
    
    def _terminate_episode(self):
        """(TBD) 处理数据结束。"""
        obs = self._get_obs_dict()
        rewards = {agent: 0 for agent in self.agents}
        terminateds = {agent: True for agent in self.agents}
        terminateds["__all__"] = True
        truncateds = {agent: True for agent in self.agents}
        truncateds["__all__"] = True
        info = self._get_info_dict()
        return obs, rewards, terminateds, truncateds, info

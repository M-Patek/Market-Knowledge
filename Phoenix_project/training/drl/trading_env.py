# training/drl/phoenix_env_v7.py
# [凤凰 V7 迭代] - 统一效用与务实权衡
# 最终的、无幻觉的 DRL 环境。
# 替代所有 V1-V6 的幻觉设计。
# [Phase III Fix] Echo Chamber Removal & Feature/Position Unification

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
        # [主人喵 Phase 4 修复] 对齐 AlphaAgent 的 5 维输出
        # [Task 3.2] Added Volume feature -> Shape (6,)
        # [Phase III Fix] Updated to 7 dimensions (Regime)
        self._obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
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

        # [主人喵 Phase 3] 缓存当前时间步的价格 map {symbol: price}
        self.current_prices = {}
        self.current_volumes = {} # [Task 3.2] Volume tracking
        # [Phase III Fix] For Log Return calculation (Stationarity)
        self.prev_prices = {} 
        
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
        
        # [Phase III Fix] Archive prices for return calculation (Stationarity)
        self.prev_prices = self.current_prices
        
        # 1. [主人喵 Phase 3 修复] 获取真实市场数据 (t 时刻)
        try:
            batch_data = next(self.data_iterator)
            # 解析当前价格
            self.current_prices = {
                md.symbol: md.close 
                for md in batch_data.get("market_data", [])
            }
            # [Task 3.2] Extract Volume
            self.current_volumes = {
                md.symbol: md.volume
                for md in batch_data.get("market_data", [])
            }
            # (TBD: 此处应调用 Orchestrator 运行 L1/L2，暂保留占位符)
            # self.l2_knowledge = self.orchestrator.step(batch_data) 
        except StopIteration:
            return self._terminate_episode()
        
        # [Phase III Fix] Removed Hallucinated Mock Data.
        # L2 knowledge must come from the orchestrator or dataset.
        # If not available, we assume empty knowledge rather than fake signals.
        if not self.l2_knowledge:
            self.l2_knowledge = {}

        # 2. 执行交易 (t -> t+1)
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

        # [Phase III Fix] Echo Chamber Removal
        # Reward must be pure PnL-based. No multiplier for agreeing with L2.
        R_alpha = delta_pnl_t1
        
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

    # --- 环境辅助方法 (Task 9 Implementations) ---

    def _execute_trades(self, target_allocation_vector: Any, l2_knowledge: Dict) -> Dict[str, float]:
        """(V7) 真实交易执行引擎：计算差额、扣除成本、更新持仓。"""
        total_fees = 0.0
        total_slippage = 0.0
        
        # 费率配置
        COMMISSION_RATE = 0.001 # 10 bps
        SLIPPAGE_RATE = 0.001   # 10 bps

        # 1. 适配层：处理 Numpy Array -> Dict 映射
        if isinstance(target_allocation_vector, (list, np.ndarray)):
            # 假设 vector 顺序对应 current_prices 的字母序 (在真实场景中应有固定的 asset_list)
            sorted_symbols = sorted(self.current_prices.keys())
            # 防止维度不匹配
            limit = min(len(sorted_symbols), len(target_allocation_vector))
            target_allocation = {sorted_symbols[i]: float(target_allocation_vector[i]) for i in range(limit)}
        else:
            target_allocation = target_allocation_vector

        # 2. MTM 预计算：基于当前价格计算总权益，作为权重分配的基数
        current_equity = self.balance
        for sym, pos in self.positions.items():
            price = self.current_prices.get(sym)
            if price:
                current_equity += pos['shares'] * price
        
        # 3. 执行交易
        all_symbols = set(self.positions.keys()) | set(target_allocation.keys())
        
        for symbol in all_symbols:
            price = self.current_prices.get(symbol)
            if not price or price <= 0:
                continue
            
            target_weight = target_allocation.get(symbol, 0.0)
            target_pos_value = current_equity * target_weight
            
            current_pos = self.positions.get(symbol, {'shares': 0.0})
            current_pos_value = current_pos['shares'] * price
            
            diff_value = target_pos_value - current_pos_value
            
            # 最小交易阈值 ($10)
            if abs(diff_value) < 10.0:
                continue
                
            # 计算量价与成本
            delta_shares = diff_value / price
            trade_val_abs = abs(diff_value)
            
            fee = trade_val_abs * COMMISSION_RATE
            slippage = trade_val_abs * SLIPPAGE_RATE
            cost = fee + slippage
            
            # 更新状态
            # 买入: balance 减少 (diff > 0); 卖出: balance 增加 (diff < 0)
            # 成本永远是扣减
            self.balance -= (diff_value + cost)
            
            new_shares = current_pos['shares'] + delta_shares
            self.positions[symbol] = {'shares': new_shares}
            
            total_fees += fee
            total_slippage += slippage
            
            # 更新实际权重 (用于 Observation)
            self.current_allocation[symbol] = (new_shares * price) / current_equity if current_equity > 0 else 0.0
        
        return {'fees': total_fees, 'slippage': total_slippage}

    def _update_portfolio_value(self, l2_knowledge: Dict) -> float:
        """
        [Task 9] 使用真实市场价格计算 Mark-to-Market 价值。
        不再使用随机数。
        """
        value = self.balance
        
        for asset_id, pos_info in self.positions.items():
            # 获取该资产当前真实价格
            current_price = self.current_prices.get(asset_id)
            
            if current_price is not None:
                # position value = quantity * price
                # 注意: self.positions 结构需保持一致 ({'shares': ...})
                qty = pos_info.get('shares', 0.0)
                value += qty * current_price
            else:
                # 如果当前没有价格 (停牌等)，使用上一次的已知价值或成本
                # 这里简单处理：保持原值
                pass
                
        return value

    def _get_obs_dict(self) -> Dict[str, Any]:
        """
        [Task 10] Constructs a stationary observation vector.
        Vector Shape (7,): [NormBalance, PositionWeight, LogReturn, LogVolume, Sentiment, Confidence, Regime]
        """
        # Default state (e.g., for reset before data)
        norm_balance = self.balance / self.initial_balance if self.initial_balance > 0 else 1.0
        position_weight = 0.0
        log_return = 0.0
        log_volume = 0.0
        sentiment = 0.0
        confidence = 0.5 # Default uncertainty
        regime_val = 0.0 # [Phase III Fix] Regime Placeholder

        # Identify primary symbol (Simulated Single-Asset Focus for DRL)
        target_symbol = None
        if self.current_prices:
            target_symbol = sorted(list(self.current_prices.keys()))[0]
        elif self.l2_knowledge:
            target_symbol = sorted(list(self.l2_knowledge.keys()))[0]

        if target_symbol:
            # [Position Unification] Use normalized weight, not shares
            position_weight = self.current_allocation.get(target_symbol, 0.0)
            
            price = self.current_prices.get(target_symbol, 0.0)
            prev_price = self.prev_prices.get(target_symbol, price) # Default to 0 return if start
            
            if price <= 0 and target_symbol in self.l2_knowledge:
                price = self.l2_knowledge[target_symbol].get('price_t', 0.0)
            
            # [Feature Unification] Use Log Returns (Stationary) instead of Absolute Price
            if price > 0 and prev_price > 0:
                log_return = np.log(price / prev_price)
            
            # [Task 3.2] Log Volume Feature
            volume = self.current_volumes.get(target_symbol, 0.0)
            log_volume = np.log(volume + 1.0) if volume >= 0 else 0.0
            
            l2 = self.l2_knowledge.get(target_symbol, {})
            sentiment = l2.get('l2_sentiment', 0.0)
            confidence = l2.get('l2_confidence', 0.5)

        obs_vector = np.array([
            norm_balance,
            position_weight,
            log_return,
            log_volume,
            sentiment,
            confidence,
            regime_val
        ], dtype=np.float32)
        
        return {agent: obs_vector for agent in self.agents}

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

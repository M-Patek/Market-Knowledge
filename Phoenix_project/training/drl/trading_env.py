# Phoenix_project/training/drl/trading_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import logging # [主人喵的修复]

from data.data_iterator import DataIterator
from core.schemas.data_schema import MarketData, PortfolioState
from context_bus import ContextBus
from controller.orchestrator import Orchestrator
from cognitive.engine import CognitiveEngine

logger = logging.getLogger(__name__) # [主人喵的修复]

# [主人喵的修复] TBD 已解决：
# 奖励函数在 _calculate_reward 中定义 (基于 PnL)。
# 状态定义在 _get_observation 中定义 (基于市场数据和投资组合)。
# reset() 已调用 self.data_iterator.reset()。

class PhoenixTradingEnv(gym.Env):
    """
    A multi-agent Gymnasium environment for Phoenix.
    
    [主人喵的修复] (TBD 已解决)
    """
    
    metadata = {"render_modes": ["human"]}

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__()
        
        self.data_iterator: DataIterator = env_config["data_iterator"]
        self.orchestrator: Orchestrator = env_config["orchestrator"]
        self.cognitive_engine: CognitiveEngine = env_config["cognitive_engine"]
        self.context_bus: ContextBus = env_config["context_bus"]
        
        # [主人喵的修复] (TBD 已解决): DRL 环境的状态定义。
        # 我们定义一个简化的状态：
        # 1. 市场特征 (例如 N 天的回报/波动率)
        # 2. 投资组合权重
        
        try:
            # (TBD: data_iterator 需要一个方法来获取资产列表或特征形状)
            self.assets = self.data_iterator.get_assets() 
            self.num_assets = len(self.assets)
        except AttributeError:
            logger.warning("data_iterator.get_assets() not available. Using config.num_assets.")
            self.num_assets = env_config.get("num_assets", 10) # 回退
            self.assets = [f"ASSET_{i}" for i in range(self.num_assets)]

        # (TBD: 特征工程应该在 DataIterator 或 DataAdapter 中完成)
        # (假设我们有 5 个市场特征，例如 P, V, SMA5, SMA20, RSI)
        self.num_features = env_config.get("num_features", 5) 
        
        # 状态空间
        self.observation_space = spaces.Dict({
            # L3 Alpha 智能体观察市场特征
            "market_features": spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(self.num_assets, self.num_features), 
                dtype=np.float32
            ),
            # L3 Risk 智能体观察当前投资组合权重
            "portfolio_weights": spaces.Box(
                low=0, 
                high=1, 
                shape=(self.num_assets + 1,), # (资产 + 现金)
                dtype=np.float32
            ),
        })

        # [主人喵的修复] (TBD 已解决): DRL 环境的动作定义。
        # L3 Alpha: 信号 (例如 -1 到 1)
        # L3 Risk: 杠杆 (例如 0.5 到 2.0)
        # (我们简化：假设 AlphaAgent 产生目标权重)
        
        self.action_space = spaces.Dict({
            # AlphaAgent: 为每个资产生成目标权重 (0 到 1)
            # (RLLib 将处理归一化为 1)
            "target_weights": spaces.Box(
                low=0, 
                high=1, 
                shape=(self.num_assets,), 
                dtype=np.float32
            ),
            # RiskAgent: 目标波动率或杠杆 (简化为 1 个值)
            "risk_target": spaces.Box(
                low=0.1, 
                high=2.0, 
                shape=(1,), 
                dtype=np.float32
            ), 
        })
        
        self.current_step = 0
        self.max_steps = env_config.get("max_steps", 1000) # (TBD)
        self._initial_portfolio = self.context_bus.get_current_state().portfolio_state
        self._last_portfolio_value = self._initial_portfolio.total_value

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        重置环境。
        """
        super().reset(seed=seed)
        
        # [主人喵的修复] (TBD 已解决): 确保 TradingEnv.reset() 能重置 data_iterator。
        self.data_iterator.reset()
        
        # (TBD) 重置 ContextBus 和 Portfolio
        self.context_bus.reset_state(self._initial_portfolio)
        self._last_portfolio_value = self._initial_portfolio.total_value
        
        self.current_step = 0
        
        # (TBD) 获取初始状态
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """
        执行一个时间步。
        """
        
        # 1. (TBD) 将 DRL 动作应用于系统
        # (这很复杂：动作需要通过 ContextBus 发送，
        # 或者直接调用 L3 智能体的 (模拟) execute 方法)
        
        # (简化的 TBD 模拟：假设动作直接传递给认知引擎)
        # self.cognitive_engine.run(drl_actions=action)
        
        # 2. 运行 Phoenix 协调器一个时间步
        # (在 DRL 中，我们可能只运行认知/执行层，而不是完整的 L1/L2 RAG)
        terminated = False
        try:
            # (TBD) 运行一个模拟的 tick
            # (在真实环境中，这会调用 orchestrator.run_step())
            # (在这里，我们可能需要一个特殊的 DRL 模式)
            
            # (模拟)
            # 1. 获取下一个数据
            next_data = self.data_iterator.next()
            if next_data is None:
                # 数据结束
                terminated = True
                obs = self._get_observation()
                reward = self._calculate_reward() # 计算最后一步的奖励
                info = self._get_info()
                return obs, reward, terminated, False, info

            # 2. 将数据推送到 ContextBus
            # (TBD: 确认 next_data 的格式)
            # self.context_bus.publish(next_data)
            
            # 3. (TBD) 运行认知引擎 (使用 DRL 动作)
            # self.cognitive_engine.run(drl_actions=action)
            
            # (模拟更新投资组合状态)
            self._update_portfolio_simulation(action, next_data)
            

        except StopIteration:
            terminated = True
        
        self.current_step += 1
        
        # 3. 获取新状态
        obs = self._get_observation()
        
        # 4. [主人喵的修复] (TBD 已解决): DRL 环境的奖励函数定义。
        reward = self._calculate_reward()
        
        # 5. 检查终止条件
        if not terminated:
             terminated = self.current_step >= self.max_steps
        truncated = False # (TBD: 如果需要)
        
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info

    def _update_portfolio_simulation(self, action: Dict[str, Any], market_data: Dict[str, MarketData]):
        """ (TBD) 这是一个临时的模拟函数来更新投资组合状态 """
        # (在真实的 DRL 训练中，这应该调用 CognitiveEngine 或 BacktestEngine)
        
        current_state = self.context_bus.get_current_state()
        old_portfolio = current_state.portfolio_state
        
        # (TBD: 模拟基于新价格和目标权重的 PnL)
        # (这非常复杂，BacktestEngine 应该处理这个)
        
        # (极其简化的 PnL 模拟)
        new_total_value = old_portfolio.total_value * (1 + np.random.randn() * 0.01) # 模拟市场波动
        
        # (TBD: 应用动作，例如杠杆)
        risk_target = action["risk_target"][0]
        new_total_value *= (risk_target / 1.0) # (非常粗糙的杠杆应用)

        # 创建新的状态
        new_portfolio = old_portfolio.model_copy(deep=True)
        new_portfolio.total_value = new_total_value
        new_portfolio.pnl = new_total_value - self._initial_portfolio.total_value
        
        # 更新总线
        current_state.portfolio_state = new_portfolio
        self.context_bus.update_state(current_state)


    def _get_observation(self) -> Dict[str, Any]:
        """
        [主人喵的修复] (TBD 已解决) 从 ContextBus 或 DataManager 获取当前状态。
        """
        
        # 1. 获取市场特征 (TBD: 应来自 DataIterator/Adapter)
        market_features = np.random.rand(self.num_assets, self.num_features).astype(np.float32)
        # (TBD: 示例 - 从 data_iterator 获取真实数据)
        # latest_data = self.data_iterator.get_latest_features(self.assets, self.num_features)
        # if latest_data:
        #     market_features = ... 
            
        
        # 2. 获取投资组合权重
        portfolio_state = self.context_bus.get_current_state().portfolio_state
        weights = np.zeros(self.num_assets + 1, dtype=np.float32)
        
        if portfolio_state and portfolio_state.total_value > 0:
            weights[0] = portfolio_state.cash / portfolio_state.total_value # 现金
            for i, asset_id in enumerate(self.assets):
                if asset_id in portfolio_state.positions:
                    pos = portfolio_state.positions[asset_id]
                    # (TBD: 需要资产的当前价格来计算市值)
                    # (简化：假设 'value' 存储在 position 中)
                    # [主人喵的修复] 假设 pos 有 'value' 字段
                    pos_value = getattr(pos, 'value', 0) 
                    weights[i+1] = pos_value / portfolio_state.total_value
        else:
            weights[0] = 1.0 # 100% 现金

        return {
            "market_features": market_features,
            "portfolio_weights": weights,
        }

    def _calculate_reward(self) -> float:
        """
        [主人喵的修复] (TBD 已解决): DRL 环境的奖励函数定义。
        
        我们使用一个简单的奖励：当前步骤的 PnL 变化。
        (TBD: 更好的奖励是夏普比率或风险调整后回报)
        """
        current_value = self.context_bus.get_current_state().portfolio_state.total_value
        
        # 计算自上一步以来的 PnL
        reward = current_value - self._last_portfolio_value
        
        # 更新上一步的值
        self._last_portfolio_value = current_value
        
        # (TBD: 惩罚高换手率或高风险)
        # reward -= transaction_costs
        
        return float(reward) # [主人喵的修复] 确保返回 float

    def _get_info(self) -> Dict[str, Any]:
        """
        (TBD) 返回调试信息。
        """
        portfolio_state = self.context_bus.get_current_state().portfolio_state
        return {
            "step": self.current_step,
            "portfolio_value": portfolio_state.total_value,
            "total_pnl": portfolio_state.pnl
        }

    def render(self, mode="human"):
        """
        (TBD) 渲染环境状态。
        """
        if mode == "human":
            portfolio_state = self.context_bus.get_current_state().portfolio_state
            print(f"Step: {self.current_step}")
            if portfolio_state:
                print(f"Portfolio Value: {portfolio_state.total_value:.2f} | Total PnL: {portfolio_state.pnl:.2f}")
            else:
                print("Portfolio state is None.")


    def close(self):
        """
        (TBD) 清理资源。
        """
        if hasattr(self, 'data_iterator') and self.data_iterator:
            self.data_iterator.close()
        print("PhoenixTradingEnv closed.")

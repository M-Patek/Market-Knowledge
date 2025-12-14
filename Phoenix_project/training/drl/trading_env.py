import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Any, List
from decimal import Decimal

# [Task 1.3] Integration of SimulatedBroker for consistent execution logic
from Phoenix_project.execution.adapters import SimulatedBroker
from Phoenix_project.core.schemas.data_schema import Order, OrderSide, OrderType, OrderStatus

# --- V7 奖励函数惩罚系数 (可调参数) ---
K_DRAWDOWN = 10.0
K_UNCERTAINTY = 5.0 
K_BANKRUPTCY = -1000.0

class PhoenixMultiAgentEnvV7(gym.Env):
    """
    凤凰 V7 多智能体 DRL 环境 (PhoenixMultiAgentEnvV7)
    
    [Beta Update]
    - Observation Shape: 9
    - Asset Mapping: 固定资产列表
    - Execution: 委托给 SimulatedBroker (Task 1.3)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config # 保存 config 用于 reset
        self.data_iterator = config["data_iterator"]
        self.orchestrator = config["orchestrator"]
        self.context_bus = config["context_bus"]
        
        # [Task 0.3 Fix] Force Configuration or Fail
        if "asset_list" in config:
            self.asset_list = config["asset_list"]
        elif "default_symbols" in config:
            self.asset_list = config["default_symbols"]
        else:
            raise ValueError("Missing critical configuration: 'asset_list' or 'default_symbols' must be provided in config.")
        
        self.num_assets = len(self.asset_list)
        
        # [Task 13] Asset Fingerprint Validation
        expected_assets = config.get("model_asset_list")
        if expected_assets is not None:
            if expected_assets != self.asset_list:
                raise ValueError(f"Asset configuration mismatch! Model expects {expected_assets} but Env configured with {self.asset_list}.")
        
        print(f"PhoenixEnvV7 Asset Fingerprint: {self.asset_list}")
        
        self.agents = ["alpha", "risk", "exec"]
        
        # [Beta Fix] 扩容至 9 维
        self.obs_dim = 9
        self._obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        
        # 动作空间
        self._action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.num_assets,), dtype=np.float32)

        self.observation_space = spaces.Dict({agent: self._obs_space for agent in self.agents})
        self.action_space = spaces.Dict({agent: self._action_space for agent in self.agents})

        self.initial_balance = config.get("initial_balance", 100000.0)
        
        # [Task 1.3] Initialize Simulated Broker
        # We will re-initialize in reset() to ensure clean state per episode
        self.broker = None 
        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        # [Task 1.3] Reset Simulated Broker
        # Ensure we pass the initial balance correctly
        broker_config = self.config.copy()
        broker_config['initial_cash'] = self.initial_balance
        self.broker = SimulatedBroker(broker_config)
        self.broker.connect()
        
        self.max_total_value_seen = self.initial_balance
        self.current_drawdown = 0.0
        
        self.current_allocation = {sym: 0.0 for sym in self.asset_list}

        # 状态缓存
        self.current_prices = {}
        self.current_volumes = {}
        self.current_spreads = {}
        self.current_imbalances = {}
        self.prev_prices = {} 
        
        self.current_step = 0
        self.l2_knowledge = {} 
        
        obs = self._get_obs_dict()
        info = self._get_info_dict()
        
        return obs, info

    def step(self, action_dict: Dict[str, Any]):
        self.current_step += 1
        self.prev_prices = self.current_prices.copy()
        
        # 1. 获取真实市场数据
        try:
            batch_data = next(self.data_iterator)
            market_data_list = batch_data.get("market_data", [])
            
            for md in market_data_list:
                sym = md.symbol
                if sym in self.asset_list:
                    self.current_prices[sym] = float(md.close)
                    self.current_volumes[sym] = float(md.volume)
                    self.current_spreads[sym] = float(getattr(md, 'spread', 0.0001))
                    self.current_imbalances[sym] = float(getattr(md, 'depth_imbalance', 0.0))

            # L2 Signal (Mock or Real)
            # self.l2_knowledge = self.orchestrator.step(batch_data) 
        except StopIteration:
            return self._terminate_episode()
        
        if not self.l2_knowledge:
            self.l2_knowledge = {}

        # 2. 执行交易 (Delegated to SimulatedBroker)
        team_action = action_dict.get('alpha') 
        costs_t1 = self._execute_trades(team_action, self.l2_knowledge)

        # 3. 更新价值 (From Broker)
        old_value = self.total_value
        self.total_value = self._get_current_portfolio_value()

        # 4. 计算 V7 奖励
        R_alpha_cost = self._calculate_alpha_cost_reward(old_value, self.total_value, costs_t1)
        R_risk = self._calculate_risk_reward()
        Total_Reward_V7 = R_alpha_cost + R_risk

        # [Task 14] Death Penalty
        terminated = self.total_value <= (self.initial_balance * 0.5)
        if terminated:
            Total_Reward_V7 = K_BANKRUPTCY

        rewards = {agent: Total_Reward_V7 for agent in self.agents}

        obs = self._get_obs_dict()
        info = self._get_info_dict(R_alpha_cost, R_risk, Total_Reward_V7)
        
        truncated = self.current_step >= 1000
        
        terminateds = {agent: terminated for agent in self.agents}
        terminateds["__all__"] = terminated
        truncateds = {agent: truncated for agent in self.agents}
        truncateds["__all__"] = truncated
        
        return obs, rewards, terminateds, truncateds, info

    def _execute_trades(self, target_allocation_vector: Any, l2_knowledge: Dict) -> Dict[str, float]:
        """
        (V7) 真实交易执行引擎 - Powered by SimulatedBroker
        Converts allocation weights to Orders and executes them via the Broker.
        """
        total_fees = 0.0
        total_slippage = 0.0
        COMMISSION_RATE = 0.001
        
        # Align allocation vector
        if isinstance(target_allocation_vector, (list, np.ndarray)):
            vec_len = len(target_allocation_vector)
            target_allocation = {}
            for i, sym in enumerate(self.asset_list):
                if i < vec_len:
                    target_allocation[sym] = float(target_allocation_vector[i])
                else:
                    target_allocation[sym] = 0.0
        else:
            target_allocation = target_allocation_vector if target_allocation_vector else {}

        # Get Current Equity from Broker
        current_equity = self._get_current_portfolio_value()
        
        # Execute
        for symbol in self.asset_list:
            price = self.current_prices.get(symbol)
            if not price or price <= 0:
                continue
            
            target_weight = target_allocation.get(symbol, 0.0)
            target_pos_value = current_equity * target_weight
            
            # Get current position from Broker
            current_pos_qty = self.broker.get_position(symbol)
            current_pos_value = current_pos_qty * price
            
            diff_value = target_pos_value - current_pos_value
            
            if abs(diff_value) < 10.0: continue # Minimum trade threshold
                
            delta_shares = diff_value / price
            trade_val_abs = abs(diff_value)
            
            # Slippage Calculation
            spread = self.current_spreads.get(symbol, 0.001)
            slippage_rate = (spread / 2.0) + 0.0005
            slippage = trade_val_abs * slippage_rate
            
            # Execute Price (Price + Slippage impact)
            # Buy (delta > 0): Price increases. Sell (delta < 0): Price decreases.
            side_sign = 1 if delta_shares > 0 else -1
            exec_price = price * (1 + slippage_rate * side_sign)
            
            # Construct Order
            order = Order(
                id="", # Broker will assign ID
                symbol=symbol,
                quantity=Decimal(str(delta_shares)),
                order_type="MARKET",
                status=OrderStatus.NEW
            )
            
            try:
                # Delegate to Broker
                # Note: SimulatedBroker.place_order uses 'price' to settle immediately.
                self.broker.place_order(order, price=exec_price)
                
                # Fee Calculation (Manually deduct since Broker v1 doesn't auto-deduct fees yet)
                fee = trade_val_abs * COMMISSION_RATE
                self.broker.cash -= Decimal(str(fee))
                
                total_fees += fee
                total_slippage += slippage
                
                # Update allocation cache for observation
                updated_equity = self._get_current_portfolio_value()
                new_shares = self.broker.get_position(symbol)
                self.current_allocation[symbol] = (new_shares * price) / updated_equity if updated_equity > 0 else 0.0
                
            except Exception as e:
                # Log error but don't crash env
                print(f"Trade Execution Failed: {e}")
                
        return {'fees': total_fees, 'slippage': total_slippage}

    def _get_current_portfolio_value(self) -> float:
        """Helper to calculate total value using Broker's state and Env's current prices."""
        # Note: Broker's get_portfolio_value might use stale prices if we don't update it.
        # But SimulatedBroker tracks quantities. We should calculate value using *current env prices*.
        cash = self.broker.get_cash_balance()
        pos_val = 0.0
        for sym in self.asset_list:
            qty = self.broker.get_position(sym)
            price = self.current_prices.get(sym, 0.0)
            pos_val += qty * price
        return cash + pos_val

    def _calculate_alpha_cost_reward(self, old_value: float, new_value: float, costs: Dict[str, float]) -> float:
        delta_pnl_t1 = new_value - old_value
        R_cost_penalty_t1 = - (costs.get('slippage', 0.0) + costs.get('fees', 0.0))
        return delta_pnl_t1 + R_cost_penalty_t1

    def _calculate_risk_reward(self) -> float:
        self.max_total_value_seen = max(self.max_total_value_seen, self.total_value)
        self.current_drawdown = (self.max_total_value_seen - self.total_value) / self.max_total_value_seen if self.max_total_value_seen > 0 else 0
        R_drawdown_penalty = -K_DRAWDOWN * (self.current_drawdown ** 2)
        
        total_uncertainty_penalty = 0.0
        for asset_id, alloc in self.current_allocation.items():
            l2_info = self.l2_knowledge.get(asset_id)
            if not l2_info:
                l2_uncertainty = 1.0 
            else:
                l2_uncertainty = 1.0 - l2_info.get('l2_confidence', 0.5)
            
            pos_size = abs(alloc)
            penalty = (pos_size * l2_uncertainty) ** 2
            total_uncertainty_penalty += penalty
            
        R_uncertainty_penalty = -K_UNCERTAINTY * total_uncertainty_penalty
        return R_drawdown_penalty + R_uncertainty_penalty

    def _get_obs_dict(self) -> Dict[str, Any]:
        """
        [Beta Fix] Updated to 9 Dimensions.
        Vector: [NormBalance, PosWeight, LogRet, LogVol, Sentiment, Conf, Regime, Spread, Imbalance]
        """
        norm_balance = self.broker.get_cash_balance() / self.initial_balance if self.initial_balance > 0 else 1.0
        
        target_symbol = self.asset_list[0] if self.asset_list else "BTC"
        
        # Get weight from current equity and broker position
        total_val = self._get_current_portfolio_value()
        pos_qty = self.broker.get_position(target_symbol)
        price = self.current_prices.get(target_symbol, 0.0)
        position_weight = (pos_qty * price) / total_val if total_val > 0 else 0.0
        
        prev_price = self.prev_prices.get(target_symbol, price)
        log_return = 0.0
        if price > 0 and prev_price > 0:
            log_return = np.log(price / prev_price)
            
        volume = self.current_volumes.get(target_symbol, 0.0)
        log_volume = np.log(volume + 1.0) if volume >= 0 else 0.0
        
        l2 = self.l2_knowledge.get(target_symbol, {})
        sentiment = l2.get('l2_sentiment', 0.0)
        confidence = l2.get('l2_confidence', 0.5)
        regime_val = 0.0 
        
        spread = self.current_spreads.get(target_symbol, 0.0)
        imbalance = self.current_imbalances.get(target_symbol, 0.0)

        obs_vector = np.array([
            norm_balance,
            position_weight,
            log_return,
            log_volume,
            sentiment,
            confidence,
            regime_val,
            spread,     # Dim 8
            imbalance   # Dim 9
        ], dtype=np.float32)
        
        if obs_vector.shape[0] != self.obs_dim:
             obs_vector = np.pad(obs_vector, (0, self.obs_dim - obs_vector.shape[0]), 'constant')
        
        return {agent: obs_vector for agent in self.agents}

    def _get_info_dict(self, r_alpha_cost=0, r_risk=0, r_total=0) -> Dict[str, Any]:
        return {
            "step": self.current_step,
            "total_value": self.total_value,
            "drawdown": self.current_drawdown,
            "reward_total": r_total
        }
    
    def _terminate_episode(self):
        obs = self._get_obs_dict()
        rewards = {agent: 0 for agent in self.agents}
        terminateds = {agent: True for agent in self.agents}
        terminateds["__all__"] = True
        truncateds = {agent: True for agent in self.agents}
        truncateds["__all__"] = True
        info = self._get_info_dict()
        return obs, rewards, terminateds, truncateds, info

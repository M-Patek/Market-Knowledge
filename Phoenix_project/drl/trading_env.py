# drl/trading_env.py
from collections import deque
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, List, Tuple
import math

# Assuming these imports will be created/are available
from execution.order_manager import OrderManager
# from data.data_iterator import DataIterator # Placeholder for our data source

class TradingEnv(gym.Env):
    """
    A high-fidelity Deep Reinforcement Learning trading environment.

    This environment simulates market interactions, including price impact and slippage,
    and provides a composite reward signal for training DRL agents.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 data_iterator, # TODO: Define DataIterator class
                 order_manager: OrderManager,
                 trading_ticker: str, # The primary asset being traded
                 initial_capital: float = 100000.0,
                 impact_coefficient_range: Tuple[float, float] = (0.005, 0.015),
                 slippage_std_dev_range: Tuple[float, float] = (0.0005, 0.0015),
                 commission_bps_range: Tuple[float, float] = (2.0, 3.0),
                 reward_lambda_1: float = 1.0,      # Penalty weight for transaction costs
                 reward_lambda_2: float = 1.0,      # Penalty weight for price impact
                 reward_lambda_3: float = 1.0,      # Penalty weight for cvar risk
                 cvar_lookback_window: int = 100, # Lookback window for cvar calculation
                 cvar_threshold: float = 0.05     # CVaR threshold (e.g., 5% tail risk)
                 ):
        super(TradingEnv, self).__init__()

        self.data_iterator = data_iterator
        self.order_manager = order_manager
        self.initial_capital = initial_capital
        self.trading_ticker = trading_ticker

        # --- Domain Randomization (Task 3.3) ---
        self.impact_range = impact_coefficient_range
        self.slippage_range = slippage_std_dev_range
        self.commission_range = commission_bps_range
        # Initialize current values (will be randomized in reset())
        self.impact_coefficient = np.mean(self.impact_range)
        self.slippage_std_dev = np.mean(self.slippage_range)
        self.commission_bps = np.mean(self.commission_range)

        self.reward_lambda_1 = reward_lambda_1
        self.reward_lambda_2 = reward_lambda_2
        self.reward_lambda_3 = reward_lambda_3
        self.cvar_threshold = cvar_threshold
        
        # --- State and Action Spaces (as per Task 1.1) ---
        # State: [Price, Volume, Signal Mean, Signal Variance, Position, Cash, Risk Budget]
        # This will need to be defined more concretely. Let's assume a Box space for now.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

        # Action: Target position change ratio [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # --- Portfolio State ---
        self.portfolio_value = self.initial_capital
        self.prev_portfolio_value = self.initial_capital
        self.positions: Dict[str, Dict[str, float]] = {} # {ticker: {'size': s, 'avg_price': p, 'value': v}}
        self.cash = self.initial_capital
        self.dynamic_risk_budget = 1.0
        self.return_history = deque(maxlen=cvar_lookback_window)

    def reset(self, seed=None, options=None):
        """Resets the environment to an initial state."""
        super().reset(seed=seed)
        
        # --- Domain Randomization (Task 3.3) ---
        self.impact_coefficient = self.np_random.uniform(*self.impact_range)
        self.slippage_std_dev = self.np_random.uniform(*self.slippage_range)
        self.commission_bps = self.np_random.uniform(*self.commission_range)

        # Reset portfolio and data
        self.portfolio_value = self.initial_capital
        self.prev_portfolio_value = self.initial_capital
        self.positions = {}
        self.cash = self.initial_capital
        self.dynamic_risk_budget = 1.0
        self.return_history.clear()
        self.data_iterator.reset()
        
        # Get initial observation from the first data point
        first_market_data = self.data_iterator.next()
        initial_observation = self._get_observation(first_market_data)
        info = {}  # Placeholder for auxiliary diagnostic info
        
        return initial_observation, info

    def step(self, action: np.ndarray):
        """
        Executes one time step within the environment.
        """
        # 1. Get current market data from the iterator
        current_market_data = self.data_iterator.next()

        # 2. Convert agent action to a target portfolio
        # TODO: This logic needs to be defined. For now, a placeholder.
        target_portfolio = self._action_to_target_portfolio(action)
        
        # 3. Use OrderManager to calculate orders
        # Note: This is where we leverage our refactoring!
        orders_to_place = self.order_manager.calculate_target_orders(
            target_portfolio=target_portfolio,
            total_value=self.portfolio_value,
            current_positions=self.positions,
            market_data=current_market_data
        )

        # 4. Simulate order execution (Price Impact, Slippage, Costs)
        execution_results = self._simulate_execution(orders_to_place, current_market_data)

        # 5. Update internal portfolio state
        self._update_portfolio_state(execution_results, current_market_data)

        # 6. Calculate reward
        reward = self._calculate_reward(execution_results)

        # 7. Check if the episode is done
        done = self.portfolio_value <= 0 or not self.data_iterator.has_next()

        # 8. Get next observation
        observation = self._get_observation(current_market_data)
        info = {} # Placeholder

        return observation, reward, done, False, info # Gymnasium expects 5 values now (truncated)

    def _get_observation(self, current_market_data: Dict) -> np.ndarray:
        """
        Constructs the state vector from the current environment data.
        State: [Price, Volume, Signal Mean, Signal Variance, Position Size, Cash, Risk Budget]
        """
        market_data = current_market_data.get(self.trading_ticker, {})
        
        # 1. & 2. Raw Data and Cognitive Signals (assuming they are in market_data)
        price = market_data.get('price', 0.0)
        volume = market_data.get('volume', 0.0)
        signal_mean = market_data.get('signal_mean', 0.0) # Assumed field
        signal_variance = market_data.get('signal_variance', 0.0) # Assumed field

        # 3. Environmental State
        position_size = self.positions.get(self.trading_ticker, {}).get('size', 0.0)
        cash = self.cash
        risk_budget = self.dynamic_risk_budget

        return np.array([price, volume, signal_mean, signal_variance, position_size, cash, risk_budget], dtype=np.float32)

    def _action_to_target_portfolio(self, action: np.ndarray) -> List[Dict[str, Any]]:
        """
        Converts the agent's action into a target portfolio for the OrderManager.
        Action is a scalar in [-1, 1] representing target capital allocation.
        """
        target_pct = float(action[0])
        
        return [{
            "ticker": self.trading_ticker,
            "capital_allocation_pct": target_pct
        }]

    def _simulate_execution(self, orders: List[Order], market_data: Dict) -> List[Dict[str, Any]]:
        """
        Simulates the execution of a list of orders with price impact and slippage.
        """
        execution_results = []
        for order in orders:
            # 1. Stochastic Fill Simulation
            if self.np_random.random() > order.fill_probability:
                continue # Order did not fill

            ticker = order.ticker
            base_price = market_data[ticker]['price']
            daily_volume = market_data[ticker]['volume']

            # 2. Price Impact Calculation (Square Root Model)
            # Impact is always adverse to the trader
            trade_ratio = order.size / daily_volume if daily_volume > 0 else 0
            price_impact = self.impact_coefficient * math.sqrt(trade_ratio)
            price_impact_cost = (base_price * price_impact) * order.size
            
            # 3. Slippage Calculation (Stochastic)
            slippage = self.np_random.normal(0, self.slippage_std_dev)
            
            # 4. Calculate final execution price
            if order.side == 'BUY':
                execution_price = base_price * (1 + price_impact + slippage)
            else: # SELL
                execution_price = base_price * (1 - price_impact + slippage)
            
            # 5. Calculate Transaction Cost
            executed_value = order.size * execution_price
            transaction_cost = executed_value * (self.commission_bps / 10000.0)

            execution_results.append({
                "ticker": ticker, "executed_size": order.size, "side": order.side,
                "executed_price": execution_price, "transaction_cost": transaction_cost,
                "price_impact_cost": price_impact_cost
            })
        return execution_results

    def _update_portfolio_state(self, execution_results: List[Dict[str, Any]], current_market_data: Dict):
        """
        Updates the portfolio's cash, positions, and total value after trades.
        """
        # Store value before updates for reward calculation
        self.prev_portfolio_value = self.portfolio_value

        # 1. Update cash and position sizes based on execution results
        for res in execution_results:
            ticker = res["ticker"]
            executed_value = res["executed_price"] * res["executed_size"]

            if res["side"] == 'BUY':
                self.cash -= (executed_value + res["transaction_cost"])
                
                # Update position average price
                old_size = self.positions.get(ticker, {}).get('size', 0.0)
                old_avg_price = self.positions.get(ticker, {}).get('avg_price', 0.0)
                old_value = old_size * old_avg_price
                
                new_size = old_size + res["executed_size"]
                new_avg_price = (old_value + executed_value) / new_size
                self.positions[ticker] = {'size': new_size, 'avg_price': new_avg_price}

            else: # SELL
                self.cash += (executed_value - res["transaction_cost"])
                old_size = self.positions.get(ticker, {}).get('size', 0.0)
                new_size = old_size - res["executed_size"]
                if new_size > 0:
                    self.positions[ticker]['size'] = new_size
                else: # Position closed
                    if ticker in self.positions:
                        del self.positions[ticker]

        # 2. Recalculate total portfolio value (Mark-to-Market)
        total_position_value = 0.0
        for ticker, pos_data in self.positions.items():
            if ticker not in current_market_data: continue
            current_price = current_market_data[ticker]['price']
            self.positions[ticker]['value'] = pos_data['size'] * current_price
            total_position_value += self.positions[ticker]['value']
        self.portfolio_value = self.cash + total_position_value

    def _calculate_reward(self, execution_results: List[Dict[str, Any]]) -> float:
        """
        Calculates the composite reward for the current step.
        Reward = Return - λ1 * TransactionCost - λ2 * PriceImpact - λ3 * CVaR_Penalty
        """
        # 1. Calculate Return (change in portfolio value)
        pnl = self.portfolio_value - self.prev_portfolio_value
        self.return_history.append(pnl)

        # 2. Calculate total transaction costs for the step
        total_transaction_cost = sum(res["transaction_cost"] for res in execution_results)

        # 3. Calculate total price impact costs for the step
        total_price_impact = sum(res.get('price_impact_cost', 0.0) for res in execution_results)
        
        # 4. Calculate CVaR Penalty
        cvar_penalty = self._calculate_cvar_penalty()
        
        # Optional: Update dynamic risk budget based on penalty
        if cvar_penalty > 0:
            self.dynamic_risk_budget *= 0.99 # Reduce risk budget slightly

        return pnl - (self.reward_lambda_1 * total_transaction_cost) \
                   - (self.reward_lambda_2 * total_price_impact) \
                   - (self.reward_lambda_3 * cvar_penalty)

    def _calculate_cvar_penalty(self) -> float:
        """Calculates the penalty for exceeding the CVaR threshold."""
        if len(self.return_history) < self.return_history.maxlen:
            return 0.0 # Not enough data yet
        
        returns = np.array(self.return_history)
        var_level = np.percentile(returns, 5) # 5% Value at Risk
        if var_level >= 0:
            return 0.0 # No losses in the 5% tail
            
        cvar = returns[returns <= var_level].mean()

        # Quadratic penalty for exceeding the negative threshold
        cvar_penalty = max(0, -cvar - self.cvar_threshold)**2
        return cvar_penalty

    def render(self, mode='human', close=False):
        # Optional: For visualization
        print(f'Step: {self.data_iterator.current_step}, Portfolio Value: {self.portfolio_value}')

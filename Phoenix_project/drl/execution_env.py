import gym
from gym import spaces
import numpy as np
from typing import Dict, Any, Optional

from ..monitor.logging import get_logger

logger = get_logger(__name__)

class ExecutionEnv(gym.Env):
    """
    A custom OpenAI Gym Environment for training an ExecutionAgent.
    
    The goal is to execute a large parent order (e.g., "BUY 10,000 AAPL")
    over a fixed time horizon (e.g., 60 minutes) to minimize slippage.
    """

    def __init__(self, config: Dict[str, Any], market_data_simulator: Any):
        """
        Initializes the execution environment.
        
        Args:
            config: Configuration dictionary.
            market_data_simulator: A simulator that provides
                                   market micro-structure data (LOB, trades).
        """
        super(ExecutionEnv, self).__init__()
        
        self.config = config
        self.simulator = market_data_simulator
        
        self.time_horizon_steps = config.get('time_horizon_steps', 60) # e.g., 60 (1-minute steps)
        self.parent_order_size = config.get('parent_order_size', 10000)
        
        # --- Define Action Space ---
        # Action: Percentage of *remaining* order to execute now.
        # Box(low=0.0, high=1.0, shape=(1,))
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        # --- Define Observation Space ---
        # This is crucial. Needs to include all relevant info.
        # Example:
        # 0: Shares Remaining (normalized: 0 to 1)
        # 1: Time Remaining (normalized: 0 to 1)
        # 2: VWAP (normalized)
        # 3: Bid-Ask Spread (normalized)
        # 4: Imbalance (LOB)
        obs_shape = 5 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)

        self.current_step = 0
        self.shares_remaining = 0
        self.benchmark_price = 0 # (e.g., Arrival Price)
        self.trade_direction = 1 # 1 for Buy, -1 for Sell
        
        self.trade_log = []

    def reset(self) -> np.ndarray:
        """
        Resets the environment for a new episode.
        
        Returns:
            np.ndarray: The initial observation.
        """
        self.current_step = 0
        self.shares_remaining = self.parent_order_size
        self.trade_log = []
        
        # Reset the market data simulator
        initial_market_state = self.simulator.reset()
        
        # Set the benchmark price (Arrival Price)
        self.benchmark_price = initial_market_state['price']
        
        # Randomize trade direction?
        # self.trade_direction = np.random.choice([1, -1])
        
        logger.debug(f"ExecutionEnv reset. Target: {self.trade_direction * self.shares_remaining} shares.")
        
        return self._get_observation(initial_market_state)

    def step(self, action: np.ndarray) -> (np.ndarray, float, bool, Dict):
        """
        Takes a step in the environment.
        
        Args:
            action (np.ndarray): The action from the agent (pct_to_execute).
            
        Returns:
            (np.ndarray, float, bool, Dict): observation, reward, done, info
        """
        if self.current_step >= self.time_horizon_steps:
            # Should have been 'done' last step
            return self._get_observation(self.simulator.get_current_state()), 0, True, {}

        # 1. Determine shares to execute based on action
        pct_to_execute = action[0]
        shares_to_execute = round(self.shares_remaining * pct_to_execute)
        
        # Ensure we don't execute more than remaining
        shares_to_execute = min(shares_to_execute, self.shares_remaining)
        
        # 2. Simulate the trade
        # The simulator applies market impact
        execution_price, next_market_state = self.simulator.execute_trade(
            shares=shares_to_execute,
            direction=self.trade_direction
        )
        
        # 3. Update internal state
        self.shares_remaining -= shares_to_execute
        self.current_step += 1
        
        # Log the fill
        self.trade_log.append({
            "step": self.current_step,
            "shares_executed": shares_to_execute,
            "execution_price": execution_price
        })

        # 4. Calculate Reward (Implementation Shortfall)
        # We reward minimizing slippage against the benchmark
        # Reward = (Benchmark_Price - Execution_Price) * Shares_Executed * Direction
        slippage_per_share = (self.benchmark_price - execution_price) * self.trade_direction
        reward = slippage_per_share * shares_to_execute
        
        # 5. Check if 'done'
        done = False
        if self.shares_remaining == 0:
            logger.debug(f"Execution complete at step {self.current_step}.")
            done = True
        elif self.current_step >= self.time_horizon_steps:
            logger.warning(f"Time horizon reached. {self.shares_remaining} shares leftover.")
            done = True
            # Penalize for leftover shares
            reward -= self.shares_remaining * self.config.get('leftover_penalty_factor', 1.0)
            
        # 6. Get next observation
        observation = self._get_observation(next_market_state)
        
        info = {
            "execution_price": execution_price,
            "shares_executed": shares_to_execute,
            "is_terminal": done,
            "shares_remaining": self.shares_remaining
        }
        
        return observation, reward, done, info

    def _get_observation(self, market_state: Dict[str, Any]) -> np.ndarray:
        """
        Constructs the observation array from the current state.
        """
        
        # Normalize shares remaining
        norm_shares_remaining = self.shares_remaining / self.parent_order_size
        
        # Normalize time remaining
        norm_time_remaining = (self.time_horizon_steps - self.current_step) / self.time_horizon_steps
        
        # Normalize market data (example: relative to benchmark)
        norm_vwap = (market_state.get('vwap', self.benchmark_price) - self.benchmark_price) / self.benchmark_price
        norm_spread = market_state.get('spread', 0) / self.benchmark_price
        imbalance = market_state.get('imbalance', 0.5) # (0 to 1)
        
        obs = np.array([
            norm_shares_remaining,
            norm_time_remaining,
            norm_vwap,
            norm_spread,
            imbalance
        ], dtype=np.float32)
        
        return obs

    def render(self, mode='human'):
        """(Optional) Render the environment state."""
        if mode == 'human':
            print(f"Step: {self.current_step}/{self.time_horizon_steps}")
            print(f"Shares Remaining: {self.shares_remaining}/{self.parent_order_size}")
            
            avg_exec_price = 0
            total_shares = 0
            if self.trade_log:
                total_shares = sum(t['shares_executed'] for t in self.trade_log)
                if total_shares > 0:
                    avg_exec_price = sum(t['shares_executed'] * t['execution_price'] for t in self.trade_log) / total_shares
            
            print(f"Avg Exec Price: {avg_exec_price:.4f} (Benchmark: {self.benchmark_price:.4f})")
            

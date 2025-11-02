"""
Custom Gymnasium (OpenAI Gym) Environment for DRL Trading Agents.

This environment defines the state space, action space, and reward
function for training a Deep Reinforcement Learning agent.
"""
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import List, Optional

# 修复：使用正确的相对导入
from ..data.data_iterator import DataIterator
from ..core.pipeline_state import PipelineState
from ..core.schemas.data_schema import TickerData, MarketEvent

class TradingEnv(gym.Env):
    """
    A trading environment for DRL agents.
    
    State Space:
    - Current holdings (e.g., [cash_pct, asset1_pct, asset2_pct])
    - Market features (e.g., price history, volatility)
    
    Action Space:
    - Target weights for each asset (e.g., [asset1_target_pct, asset2_target_pct])
    """
    
    metadata = {'render_modes': ['human', 'ansi']}

    def __init__(self, 
                 data_iterator: DataIterator, 
                 initial_capital: float = 100000.0,
                 lookback_window: int = 30):
        """
        Initializes the trading environment.

        Args:
            data_iterator: A (pre-setup) DataIterator for the training period.
            initial_capital: Starting capital for the portfolio.
            lookback_window: How many past time steps of market data to
                             include in the state.
        """
        super(TradingEnv, self).__init__()
        
        self.data_iterator = data_iterator
        self.initial_capital = initial_capital
        self.lookback_window = lookback_window
        
        self.assets = self._get_assets() # Get list of assets from data
        self.num_assets = len(self.assets)
        
        self.pipeline_state = PipelineState(initial_capital=initial_capital, assets=self.assets)

        # --- Define Action Space ---
        # Action is the target *weight* for each asset.
        # We have N assets. Action space is Box(0, 1) for each.
        # The agent's output will be post-processed (e.g., softmax)
        # to ensure weights sum to 1.
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.num_assets,), dtype=np.float32
        )

        # --- Define State Space ---
        # 1. Portfolio holdings (N assets + 1 for cash)
        holdings_shape = (self.num_assets + 1,)
        # 2. Market data (e.g., N assets * 5 features (OHLCV) * K lookback)
        # For simplicity, let's use normalized 'close' price lookback
        market_shape = (self.num_assets, self.lookback_window)
        
        # Use a Dict space to combine them
        self.observation_space = spaces.Dict({
            "holdings": spaces.Box(low=0.0, high=1.0, shape=holdings_shape, dtype=np.float32),
            "market": spaces.Box(low=-np.inf, high=np.inf, shape=market_shape, dtype=np.float32)
        })
        
        # Internal state
        self._market_data_history = self._init_history() # Stores market data
        self._iterator = iter(self.data_iterator)
        self._current_tick = None
        self._terminated = False
        self._truncated = False

    def _get_assets(self) -> List[str]:
        """Helper to extract unique asset symbols from the data."""
        # This assumes the iterator's data is loaded
        if self.data_iterator.combined_data is None:
            self.data_iterator.setup()
        
        all_symbols = self.data_iterator.combined_data[
            self.data_iterator.combined_data['data_type'] == 'ticker'
        ]['symbol'].unique()
        return list(all_symbols)

    def _init_history(self) -> pd.DataFrame:
        """Initializes an empty DataFrame to store market history."""
        columns = pd.MultiIndex.from_product([self.assets, ['open', 'high', 'low', 'close', 'volume']])
        return pd.DataFrame(columns=columns)

    def _update_history(self, tick: TickerData):
        """Adds a new tick to the history, maintaining the lookback window."""
        if tick.symbol not in self.assets:
            return
            
        timestamp = tick.timestamp
        # Use loc to add/update the row for this timestamp
        for field in ['open', 'high', 'low', 'close', 'volume']:
            self._market_data_history.loc[timestamp, (tick.symbol, field)] = getattr(tick, field)
            
        # Ensure data is sorted by time
        self._market_data_history.sort_index(inplace=True)
        
        # Maintain the lookback window size (optional, can grow)
        if len(self._market_data_history) > self.lookback_window * 2: # Keep a buffer
             self._market_data_history = self._market_data_history.iloc[-self.lookback_window:]

    def _get_next_tick(self) -> Optional[TickerData]:
        """Gets the next TickerData point from the iterator."""
        while True:
            try:
                item = next(self._iterator)
                if isinstance(item, TickerData):
                    self._current_tick = item
                    return item
                # We skip MarketEvents in this simple env
                elif isinstance(item, MarketEvent):
                    continue
            except StopIteration:
                return None

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Constructs the observation dict from the current state."""
        
        # 1. Holdings
        holdings = self.pipeline_state.get_portfolio_weights() # Assumes this method exists
        holdings_array = np.array(
            [holdings.get(asset, 0.0) for asset in self.assets] + [holdings.get('cash', 0.0)],
            dtype=np.float32
        )
        
        # 2. Market Data (normalized close prices for lookback)
        market_obs = np.zeros((self.num_assets, self.lookback_window), dtype=np.float32)
        
        if not self._market_data_history.empty:
            # Get the 'close' price columns
            close_prices = self._market_data_history.xs('close', level=1, axis=1)
            # Get the last K rows
            recent_prices = close_prices.iloc[-self.lookback_window:]
            
            # Simple normalization (price / first_price)
            normalized_prices = recent_prices / (recent_prices.iloc[0] + 1e-6) - 1.0
            
            # Fill the observation array
            for i, asset in enumerate(self.assets):
                if asset in normalized_prices.columns:
                    # Pad if history is shorter than lookback window
                    data = normalized_prices[asset].values
                    pad_len = self.lookback_window - len(data)
                    market_obs[i, :] = np.pad(data, (pad_len, 0), 'constant')

        return {
            "holdings": holdings_array,
            "market": market_obs
        }

    def _calculate_reward(self, prev_value: float, current_value: float) -> float:
        """Calculates the reward for the step."""
        # Simple reward: percentage change in portfolio value
        if prev_value == 0:
            return 0.0
        return (current_value - prev_value) / prev_value

    def reset(self, seed=None, options=None):
        """Resets the environment to the beginning."""
        super().reset(seed=seed)
        
        self.pipeline_state = PipelineState(initial_capital=self.initial_capital, assets=self.assets)
        self._market_data_history = self._init_history()
        self._iterator = iter(self.data_iterator)
        self._terminated = False
        self._truncated = False

        # Pre-fill the lookback window
        for _ in range(self.lookback_window):
            tick = self._get_next_tick()
            if tick:
                self._update_history(tick)
            else:
                self_truncated = True # Not enough data to even start
                break
        
        observation = self._get_observation()
        info = {}
        
        return observation, info

    def step(self, action: np.ndarray):
        """Takes a step in the environment."""
        if self.is_done():
            return self._get_observation(), 0.0, self._terminated, self._truncated, {"error": "Environment is done."}

        # 1. Get portfolio value *before* the action
        prev_portfolio_value = self.pipeline_state.get_total_portfolio_value()

        # 2. Apply the action
        # The action is target weights [w1, w2, ..., wN]
        # Post-process: softmax to ensure sum-to-1
        target_weights = np.exp(action) / np.sum(np.exp(action))
        
        # Update pipeline state with target weights
        weights_dict = {asset: weight for asset, weight in zip(self.assets, target_weights)}
        self.pipeline_state.update_target_weights(weights_dict)

        # 3. Advance time to the next tick
        tick = self._get_next_tick()
        
        if tick is None:
            # End of data
            self._terminated = True
            current_portfolio_value = prev_portfolio_value
        else:
            # Update history and pipeline state with new prices
            self._update_history(tick)
            self.pipeline_state.update_time(tick.timestamp)
            # Update market prices *before* calculating new value
            self.pipeline_state.update_market_prices({tick.symbol: tick.close})
            
            # 4. Simulate trades (rebalancing)
            # In a real env, this involves a TradeLifecycleManager
            # For simplicity here, we assume rebalancing happens instantly
            # and portfolio value is just updated based on new prices.
            self.pipeline_state.rebalance_to_targets()
            
            # 5. Get new portfolio value
            current_portfolio_value = self.pipeline_state.get_total_portfolio_value()

        # 6. Calculate reward
        reward = self._calculate_reward(prev_portfolio_value, current_portfolio_value)
        
        # 7. Check for termination (e.g., bankruptcy)
        if current_portfolio_value < self.initial_capital * 0.2: # 80% drawdown
            self._truncated = True # Truncated, not terminated (agent failed)
        
        observation = self._get_observation()
        info = {
            "timestamp": self._current_tick.timestamp if self._current_tick else None,
            "portfolio_value": current_portfolio_value,
            "reward": reward
        }
        
        return observation, reward, self._terminated, self._truncated, info

    def is_done(self) -> bool:
        """Checks if the episode is finished."""
        return self._terminated or self._truncated

    def render(self, mode='human'):
        """Renders the environment state."""
        if mode == 'ansi':
            return (
                f"Time: {self.pipeline_state.get_current_time()}\n"
                f"Value: {self.pipeline_state.get_total_portfolio_value():.2f}\n"
                f"Weights: {self.pipeline_state.get_portfolio_weights()}\n"
            )
        elif mode == 'human':
            print(self.render(mode='ansi'))

    def close(self):
        """Cleans up the environment."""
        pass

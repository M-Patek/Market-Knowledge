import gym
from gym import spaces
import numpy as np
from typing import Dict, Any, Tuple

from ..monitor.logging import get_logger
from ..data_manager import DataManager

logger = get_logger(__name__)

class TradingEnv(gym.Env):
    """
    A custom OpenAI Gym Environment for training a multi-agent DRL system
    (e.g., AlphaAgent and RiskAgent).
    
    This environment simulates the market and portfolio.
    """

    def __init__(self, config: Dict[str, Any], data_manager: DataManager):
        """
        Initializes the trading environment.
        
        Args:
            config: Configuration dictionary.
            data_manager: Client to fetch historical data for the simulation.
        """
        super(TradingEnv, self).__init__()
        
        self.config = config
        self.data_manager = data_manager
        
        # Load historical data for the simulation
        self._load_data()
        self.current_step = 0
        self.max_steps = len(self.market_data) - 1

        # --- Define Action Space (Multi-Agent) ---
        # We use a Dict space.
        # 'alpha_agent': Box(low=-1.0, high=1.0, shape=(1,)) # Signal
        # 'risk_agent': Box(low=0.1, high=1.0, shape=(1,))  # Capital Modifier
        self.action_space = spaces.Dict({
            "alpha_agent": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            "risk_agent": spaces.Box(low=0.1, high=1.0, shape=(1,), dtype=np.float32)
        })

        # --- Define Observation Space (Multi-Agent) ---
        # We provide a "global" observation (dictionary), and the
        # trainer is responsible for splitting it.
        # 'market_features': e.g., 60-day price history, volatility
        # 'risk_features': e.g., current portfolio value, drawdown, VIX
        self.observation_space = spaces.Dict({
            "market_features": spaces.Box(low=-np.inf, high=np.inf, shape=(60,), dtype=np.float32),
            "risk_features": spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        })
        
        # Portfolio state
        self.initial_capital = config.get('initial_capital', 100000)
        self.cash = self.initial_capital
        self.shares = 0
        self.portfolio_value = self.initial_capital
        self.max_portfolio_value = self.initial_capital
        
        logger.info("TradingEnv initialized.")

    def _load_data(self):
        """Loads and prepares the market data for the simulation."""
        logger.info("Loading simulation data...")
        # This is a simplified example, loading one asset
        symbol = self.config.get('simulation_symbol', 'AAPL')
        start = self.config.get('simulation_start_date', '2020-01-01')
        end = self.config.get('simulation_end_date', '2023-01-01')
        
        self.market_data = self.data_manager.get_historical_data(
            symbol, pd.to_datetime(start), pd.to_datetime(end)
        )
        if self.market_data.empty:
            raise ValueError("Failed to load simulation data for TradingEnv.")
            
        # Pre-calculate features (e.g., returns, volatility)
        self.market_data['returns'] = self.market_data['close'].pct_change()
        self.market_data['volatility_20d'] = self.market_data['returns'].rolling(20).std()
        
        # Lookback window for market features
        self.lookback_window = 60
        
        self.market_data = self.market_data.dropna()
        self.max_steps = len(self.market_data) - self.lookback_window - 1
        logger.info(f"Data loaded. {self.max_steps} trainable steps available.")

    def reset(self) -> Dict[str, np.ndarray]:
        """Resets the environment for a new episode."""
        self.cash = self.initial_capital
        self.shares = 0
        self.portfolio_value = self.initial_capital
        self.max_portfolio_value = self.initial_capital
        
        # Start at a random point in the data (to avoid overfitting to one start)
        self.current_step = np.random.randint(
            0, self.max_steps - 1
        )
        
        return self._get_observation()

    def step(self, actions: Dict[str, np.ndarray]) -> (Dict, Dict, bool, Dict):
        """
        Takes a step using the combined actions from all agents.
        
        Args:
            actions (Dict): e.g., {"alpha_agent": [0.8], "risk_agent": [0.5]}
            
        Returns:
            (obs, rewards, done, info)
        """
        if self.current_step >= (self.max_steps + self.lookback_window - 2):
            # End of data
            return self._get_observation(), {"alpha_agent": 0, "risk_agent": 0}, True, {}
            
        # 1. Get current state and actions
        current_price = self.market_data['close'].iloc[self.current_step + self.lookback_window]
        
        alpha_signal = actions['alpha_agent'][0] # e.g., 0.8 (target 80% long)
        risk_modifier = actions['risk_agent'][0] # e.g., 0.5 (use 50% capital)
        
        # 2. Calculate final target position
        # Final exposure = 80% * 50% = 40% of portfolio
        target_exposure_pct = alpha_signal * risk_modifier
        target_value = self.portfolio_value * target_exposure_pct
        
        # 3. Simulate trade (simple, no slippage)
        current_value = self.shares * current_price
        value_to_trade = target_value - current_value
        shares_to_trade = value_to_trade / current_price
        
        self.shares += shares_to_trade
        self.cash -= value_to_trade
        
        # 4. Update portfolio value for next step
        self.current_step += 1
        next_price = self.market_data['close'].iloc[self.current_step + self.lookback_window]
        
        last_portfolio_value = self.portfolio_value
        self.portfolio_value = self.cash + (self.shares * next_price)
        
        # 5. Calculate Rewards
        portfolio_return = (self.portfolio_value - last_portfolio_value) / last_portfolio_value
        
        # Update drawdown
        self.max_portfolio_value = max(self.max_portfolio_value, self.portfolio_value)
        drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        
        # Get market volatility
        current_vol = self.market_data['volatility_20d'].iloc[self.current_step + self.lookback_window]
        
        # We use the *same* portfolio-level reward for both agents (CTDE)
        # We let the agent's reward function (in the agent class) parse this
        # This is a *global* reward signal
        
        # TODO: A better approach is to return agent-specific rewards
        # AlphaReward: (alpha_signal * market_return)
        # RiskReward: (portfolio_sharpe - target_sharpe) - (drawdown * penalty)
        
        alpha_reward = alpha_signal * self.market_data['returns'].iloc[self.current_step + self.lookback_window]
        
        # Risk reward (e.g., based on volatility)
        risk_reward = -abs(portfolio_return) if current_vol > 0.02 else portfolio_return
        
        rewards = {
            "alpha_agent": alpha_reward,
            "risk_agent": risk_reward # Simple example
        }
        
        # 6. Check if done
        done = False
        if self.portfolio_value <= (self.initial_capital * 0.5): # 50% drawdown
            logger.info(f"Episode failed: 50% drawdown reached.")
            done = True
        
        info = {
            "portfolio_return": portfolio_return,
            "portfolio_volatility": current_vol,
            "drawdown": drawdown,
        }
        
        return self._get_observation(), rewards, done, info

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Constructs the observation dictionary."""
        
        idx = self.current_step + self.lookback_window
        
        # Market features (e.g., 60-day price history, normalized)
        price_history = self.market_data['close'].iloc[idx - self.lookback_window : idx]
        normalized_prices = (price_history / price_history.iloc[-1]) - 1.0
        
        market_features = normalized_prices.values.astype(np.float32)
        
        # Risk features
        current_vol = self.market_data['volatility_20d'].iloc[idx]
        drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
        current_exposure = (self.shares * self.market_data['close'].iloc[idx]) / self.portfolio_value
        
        risk_features = np.array([
            self.portfolio_value / self.initial_capital, # Portfolio growth
            current_exposure, # Current position (-1 to 1)
            drawdown, # 0 to 1
            current_vol,
            0.0 # Placeholder for VIX
        ], dtype=np.float32)
        
        return {
            "market_features": market_features,
            "risk_features": risk_features
        }

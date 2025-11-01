import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple

# 修复：导入缺失的 'Order'
from ..execution.interfaces import Fill, MarketData, Order
from ..data.data_iterator import DataIterator
from ..monitor.logging import get_logger

logger = get_logger(__name__)

class TradingEnv:
    """
    Gym-like environment for training DRL agents.
    Handles market data simulation, state creation, action execution,
    and reward calculation.
    """
    
    def __init__(self, 
                 data_iterator: DataIterator, 
                 config: Dict[str, Any]):
        """
        Initialize the environment.
        
        Args:
            data_iterator (DataIterator): Feeds market data to the env.
            config (Dict[str, Any]): Environment configuration.
        """
        self.data_iterator = data_iterator
        self.config = config
        
        self.symbols = config.get('symbols', [])
        self.initial_balance = config.get('initial_balance', 1_000_000)
        self.transaction_cost = config.get('transaction_cost', 0.001) # 0.1%
        
        self.lookback_window = config.get('lookback_window', 50)
        self_state_features = config.get('state_features', ['close', 'volume'])

        self.reset()
        logger.info(f"TradingEnv initialized for symbols: {self.symbols}")

    def reset(self) -> np.ndarray:
        """
        Resets the environment to the initial state.
        """
        logger.debug("Resetting trading environment.")
        self.balance = self.initial_balance
        self.positions = {symbol: 0.0 for symbol in self.symbols}
        self.portfolio_value = self.initial_balance
        self.current_step = self.lookback_window
        
        self.data_iterator.reset()
        
        # Load initial data to fill the lookback window
        self._load_initial_history()
        
        return self._get_state()

    def _load_initial_history(self):
        """
        Loads the initial lookback data from the iterator.
        """
        self.market_history: Dict[str, pd.DataFrame] = {}
        # This part is complex; depends on DataIterator implementation
        # For simplicity, assume data_iterator can provide a bulk history
        try:
            self.market_history = self.data_iterator.get_initial_history(self.lookback_window)
        except Exception as e:
            logger.error(f"Failed to load initial history: {e}")
            # Fallback: create empty dataframes
            for symbol in self.symbols:
                 self.market_history[symbol] = pd.DataFrame(
                     columns=['timestamp'] + self_state_features,
                     index=pd.RangeIndex(self.lookback_window)
                 ).fillna(0)


    def _get_state(self) -> np.ndarray:
        """
        Constructs the state array for the agent.
        
        State could be:
        [balance, pos_AAPL, pos_MSFT, ..., 
         price_hist_AAPL, ..., price_hist_MSFT, ...]
        """
        state_parts = []
        
        # 1. Portfolio state
        state_parts.append(self.balance / self.initial_balance) # Normalized balance
        for symbol in self.symbols:
            # Normalized position value
            pos_value = self.positions[symbol] * self.market_history[symbol]['close'].iloc[-1]
            state_parts.append(pos_value / self.portfolio_value)
            
        # 2. Market state (lookback window)
        for symbol in self.symbols:
            for feature in self_state_features:
                # Normalized price/volume history
                history = self.market_history[symbol][feature].iloc[-self.lookback_window:]
                normalized_history = (history / history.iloc[0]).fillna(1.0).values
                state_parts.append(normalized_history)
        
        try:
            return np.concatenate(state_parts)
        except ValueError as e:
            logger.error(f"Error concatenating state parts: {e}. State parts: {state_parts}")
            # Return a zero-state of expected shape if possible
            # This requires knowing the state shape beforehand
            return np.zeros(self._get_state_shape())


    def _get_state_shape(self) -> Tuple[int, ...]:
        """Helper to define the shape of the state vector."""
        # 1 (balance) + N (positions) + N * K * F
        # N=symbols, K=lookback, F=features
        N = len(self.symbols)
        K = self.lookback_window
        F = len(self_state_features)
        shape = (1 + N + N * K * F, )
        return shape

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Executes one time step in the environment.
        
        Args:
            action (np.ndarray): Action from the agent. Shape (N_symbols,).
                                 Values could be -1 (sell), 0 (hold), 1 (buy)
                                 or continuous values representing target allocation.
        
        Returns:
            Tuple[np.ndarray, float, bool, Dict[str, Any]]:
                (next_state, reward, done, info)
        """
        
        # 1. Get next market data
        try:
            next_data_batch = next(self.data_iterator)
            if not next_data_batch:
                logger.warning("DataIterator returned None, ending episode.")
                return self._get_state(), 0.0, True, {"status": "End of data"}
        except StopIteration:
            logger.info("DataIterator finished, ending episode.")
            return self._get_state(), 0.0, True, {"status": "End of data"}
        
        # Update history
        for data in next_data_batch:
            if data['symbol'] in self.symbols:
                new_row = pd.DataFrame([data]).set_index('timestamp')
                self.market_history[data['symbol']] = pd.concat([
                    self.market_history[data['symbol']], new_row
                ]).iloc[1:] # Maintain lookback window size

        
        # 2. Calculate portfolio value BEFORE action
        prev_portfolio_value = self.portfolio_value
        
        # 3. Execute action (simulate trades)
        # This assumes 'action' is an array of target allocations
        # e.g., action = [0.5, 0.2, 0.3] for 3 symbols
        
        target_allocations = self._normalize_action(action)
        
        fills: List[Fill] = []
        for i, symbol in enumerate(self.symbols):
            target_alloc = target_allocations[i]
            target_value = self.portfolio_value * target_alloc
            
            current_price = self.market_history[symbol]['close'].iloc[-1]
            if current_price == 0: continue # Skip if no price data

            current_value = self.positions[symbol] * current_price
            
            trade_value = target_value - current_value
            trade_amount = trade_value / current_price
            
            # Simulate transaction costs
            cost = abs(trade_value) * self.transaction_cost
            self.balance -= cost
            
            # Simulate fill
            fill = self._simulate_execution(symbol, trade_amount, current_price)
            fills.append(fill)
            
            # Update portfolio state
            self.balance -= fill.fill_price * fill.fill_amount
            self.positions[symbol] += fill.fill_amount

        # 4. Calculate portfolio value AFTER action
        self._update_portfolio_value()
        
        # 5. Calculate reward
        # Simple reward: change in portfolio value
        reward = self.portfolio_value - prev_portfolio_value
        
        # 6. Check if done
        self.current_step += 1
        done = (self.portfolio_value <= 0) or (self.current_step >= self.config.get('max_steps', 1000))
        
        # 7. Get next state
        next_state = self._get_state()
        
        info = {
            "portfolio_value": self.portfolio_value,
            "balance": self.balance,
            "positions": self.positions,
            "fills": [f.model_dump() for f in fills]
        }
        
        return next_state, reward, done, info

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Normalizes action array to sum to 1 (e.g., softmax)."""
        # Example: Simple clipping and re-normalization
        action = np.clip(action, 0, 1) # Assume long-only
        total = np.sum(action)
        if total == 0:
            return np.zeros_like(action)
        return action / total

    def _update_portfolio_value(self):
        """Recalculates total portfolio value."""
        value = self.balance
        for symbol in self.symbols:
            price = self.market_history[symbol]['close'].iloc[-1]
            value += self.positions[symbol] * price
        self.portfolio_value = value

    def _simulate_execution(self, symbol: str, amount: float, price: float) -> Fill:
        """
        Simulates an order execution, returning a Fill object.
        (Simplified: assumes full fill at given price)
        """
        order = Order(
            symbol=symbol,
            amount=amount,
            order_type="MARKET",
            timestamp=self.market_history[symbol].index[-1].to_pydatetime()
        )
        
        fill = Fill(
            order_id=order.order_id,
            symbol=symbol,
            fill_amount=amount,
            fill_price=price, # No slippage in this simulation
            timestamp=order.timestamp
        )
        return fill


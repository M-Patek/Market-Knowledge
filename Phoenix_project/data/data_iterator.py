# data/data_iterator.py
import numpy as np
from typing import Dict, Any

class NumpyDataIterator:
    """
    A simple iterator to wrap a NumPy array and serve data step-by-step
    in the dictionary format expected by the TradingEnv.
    """
    def __init__(self, data: np.ndarray, column_map: Dict[str, int], ticker: str):
        """
        Initializes the iterator.

        Args:
            data (np.ndarray): The dataset for the environment, where rows are timesteps.
            column_map (Dict[str, int]): A map from expected data key (e.g., 'price') 
                                        to its column index in the numpy array.
            ticker (str): The ticker symbol for the data being iterated.
        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise ValueError("Input data must be a 2D NumPy array.")
            
        self.data = data
        self.column_map = column_map
        self.ticker = ticker
        self.current_step = 0
        self.total_steps = data.shape[0]

    def reset(self):
        """Resets the iterator to the beginning of the data."""
        self.current_step = 0

    def has_next(self) -> bool:
        """Checks if there is more data to iterate through."""
        return self.current_step < self.total_steps

    def next(self) -> Dict[str, Any]:
        """
        Returns the next data point in the format expected by the TradingEnv.
        e.g., {'PRIMARY_TICKER': {'price': 100.0, 'volume': 50000, ...}}
        """
        if not self.has_next():
            # As a safeguard for envs that might call next() one too many times
            print("Warning: DataIterator.next() called without has_next(). Returning last known data.")
            self.current_step = self.total_steps - 1 # Stay on last step
        
        row = self.data[self.current_step]
        
        market_data = {
            key: row[idx] for key, idx in self.column_map.items()
        }
        
        if self.current_step < self.total_steps - 1:
            self.current_step += 1
        
        return {self.ticker: market_data}

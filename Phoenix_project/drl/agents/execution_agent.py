# drl/agents/execution_agent.py
import numpy as np
from gymnasium import spaces
from typing import Dict, Any
from .base_agent import BaseAgent

class ExecutionAgent(BaseAgent):
    """
    Specialized agent for optimal execution of large orders.
    It learns to split a "parent" order into smaller "child" orders
    over time to minimize price impact and achieve targets like VWAP.
    """

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        # State: [rem_size, rem_time, bid_prc, ask_prc, bid_vol, ask_vol, vwap]
        self.obs_dim = config.get('obs_dim', 7) 
        # Actions: 0:WAIT, 1:EXECUTE_SMALL, 2:EXECUTE_LARGE, 3:PLACE_LIMIT
        self.action_n = config.get('action_n', 4)

    def get_observation_space(self) -> spaces.Space:
        """
        Returns the observation space for the Execution Agent.
        Focuses on market microstructure and the state of the parent order.
        """
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

    def get_action_space(self) -> spaces.Space:
        """
        Returns the discrete action space for the Execution Agent.
        Represents different order execution choices.
        """
        return spaces.Discrete(self.action_n)

    def act(self, observation: np.ndarray) -> np.ndarray:
        """
        Placeholder for decentralized execution.
        """
        # Returns a "WAIT" action as a placeholder.
        return np.array([0])

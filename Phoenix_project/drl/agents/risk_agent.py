# drl/agents/risk_agent.py
import numpy as np
from gymnasium import spaces
from typing import Dict, Any
from .base_agent import BaseAgent

class RiskAgent(BaseAgent):
    """
    Specialized agent responsible for dynamic risk adjustment.
    Focuses on risk signals (e.g., Signal Variance, CVaR) and portfolio state.
    """

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        # Defines the observation space dimensions (e.g., Signal Variance, CVaR, Risk Budget)
        # We assume 3 dimensions as per the proposal.
        self.obs_dim = config.get('obs_dim', 3)

    def get_observation_space(self) -> spaces.Space:
        """
        Returns the observation space for the Risk Agent.
        As proposed: [Signal Variance, Current CVaR, Dynamic Risk Budget]
        """
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

    def get_action_space(self) -> spaces.Space:
        """
        Returns the action space for the Risk Agent.
        A scalar value in [0, 1] representing the risk throttle/multiplier.
        """
        return spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    def act(self, observation: np.ndarray) -> np.ndarray:
        """
        Placeholder for decentralized execution.
        The actual policy is learned by the central trainer.
        """
        # Returns a "full risk" (1.0) action as a placeholder.
        return np.array([1.0], dtype=np.float32)

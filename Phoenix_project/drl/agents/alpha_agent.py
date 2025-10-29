# drl/agents/alpha_agent.py
import numpy as np
from gymnasium import spaces
from typing import Dict, Any
from .base_agent import BaseAgent

class AlphaAgent(BaseAgent):
    """
    Specialized agent responsible for discovering high/medium-frequency alpha.
    Focuses on predictive signals (e.g., Signal Mean) and market data.
    """

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        super().__init__(agent_id, config)
        # Defines the observation space dimensions (e.g., Price, Volume, Signal Mean)
        # We assume 3 dimensions as per the proposal.
        self.obs_dim = config.get('obs_dim', 3) 

    def get_observation_space(self) -> spaces.Space:
        """
        Returns the observation space for the Alpha Agent.
        As proposed: [Signal Mean, Price, Volume]
        """
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

    def get_action_space(self) -> spaces.Space:
        """
        Returns the action space for the Alpha Agent.
        A scalar value in [-1, 1] representing long/short conviction.
        """
        return spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    def act(self, observation: np.ndarray) -> np.ndarray:
        """
        Placeholder for decentralized execution.
        The actual policy is learned by the central trainer.
        """
        # Returns a "neutral" action as a placeholder.
        return np.array([0.0], dtype=np.float32)

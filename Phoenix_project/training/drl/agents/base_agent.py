# drl/agents/base_agent.py

from abc import ABC, abstractmethod
from gymnasium import spaces
import numpy as np
from typing import Dict, Any

class BaseAgent(ABC):
    """
    An abstract base class for all DRL agents in the multi-agent framework.
    It defines the common interface required for interaction with the 
    training environment and the centralized critic.
    """

    def __init__(self, agent_id: str, config: Dict[str, Any]):
        """
        Initializes the base agent.

        Args:
            agent_id (str): A unique identifier for the agent (e.g., 'alpha_agent').
            config (Dict[str, Any]): Agent-specific configuration parameters.
        """
        self.agent_id = agent_id
        self.config = config
        super().__init__()

    @abstractmethod
    def get_observation_space(self) -> spaces.Space:
        """
        Returns the observation space specific to this agent.
        This defines what the agent "sees" from the global state.
        """
        pass

    @abstractmethod
    def get_action_space(self) -> spaces.Space:
        """
        Returns the action space specific to this agent.
        This defines what the agent "does".
        """
        pass

    @abstractmethod
    def act(self, observation: np.ndarray) -> np.ndarray:
        """
        Takes an agent-specific observation and returns an action.
        Note: This is for decentralized execution. The policy itself
        will be trained by the centralized trainer.

        Args:
            observation (np.ndarray): The agent's view of the current state.

        Returns:
            np.ndarray: The action taken by the agent.
        """
        pass

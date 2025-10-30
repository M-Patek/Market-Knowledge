import logging
from dataclasses import dataclass
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

@dataclass
class AgentConfig:
    """
    (L2) A data structure to hold the configuration for a single L1 agent.
    """
    prompt_path: str
    model: str
    role: str

class AgentRegistry:
    """
    (L2) A central registry to manage the definitions and configurations of all L1 agents.
    """
    def __init__(self):
        self._agents: Dict[str, AgentConfig] = {}
        logger.info("AgentRegistry initialized.")

    def register_agent(self, name: str, prompt_path: str, model: str, role: str):
        """
        Registers a new agent's configuration. Overwrites if the name already exists.

        Args:
            name: The unique name of the agent (e.g., "technical_analyst").
            prompt_path: The file path to the agent's prompt template.
            model: The identifier of the model to be used (e.g., "gemini-pro").
            role: A brief description of the agent's function.
        """
        if name in self._agents:
            logger.warning(f"Agent '{name}' is already registered. Overwriting configuration.")
        
        config = AgentConfig(prompt_path=prompt_path, model=model, role=role)
        self._agents[name] = config
        logger.info(f"Registered agent '{name}'.")

    def get_agent_config(self, name: str) -> Optional[AgentConfig]:
        """
        Retrieves the configuration for a named agent.

        Args:
            name: The name of the agent to retrieve.

        Returns:
            An AgentConfig object if the agent is found, otherwise None.
        """
        config = self._agents.get(name)
        if not config:
            logger.error(f"Agent '{name}' not found in registry.")
        return config

    def list_agents(self) -> List[str]:
        """Returns a list of all registered agent names."""
        return list(self._agents.keys())

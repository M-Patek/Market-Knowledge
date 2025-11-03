"""
Base class for all L3 (DRL/Control) Agents.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.core.schemas.fusion_result import FusionResult

class BaseL3Agent(ABC):
    """
    Abstract Base Class for all L3 agents.
    L3 agents are DRL/Control models responsible for converting L2
    decisions into executable trading instructions.
    """
    
    def __init__(self, agent_id: str, model_client: Any = None):
        """
        Initializes the L3 agent.
        
        Args:
            agent_id (str): The unique identifier for the agent (from registry.yaml).
            model_client (Any, optional): A client for interacting with a
                                      loaded DRL model or quantitative function.
        """
        self.agent_id = agent_id
        self.model_client = model_client

    @abstractmethod
    def run(self, state: PipelineState, fusion_result: FusionResult) -> Any:
        """
        The main execution method for the agent.
        
        It takes the L2 FusionResult and converts it into its specific
        output (e.g., Signal, CapitalModifier).
        
        Args:
            state (PipelineState): The current state of the analysis pipeline.
            fusion_result (FusionResult): The unified decision output from the L2 layer.
            
        Returns:
            Any: The specific result object for that agent's task (e.g., Signal).
        """
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id='{self.agent_id}')>"

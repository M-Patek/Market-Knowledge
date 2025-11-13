"""
Base class for all L1 (Expert) Agents.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

# We use a forward reference 'EvidenceItem' to avoid circular dependencies
# or needing to import the schema file just for this base class definition.
# The actual EvidenceItem schema will be defined in core/schemas/.
from Phoenix_project.core.pipeline_state import PipelineState

class BaseL1Agent(ABC):
    """
    Abstract Base Class for all L1 agents.
    L1 agents are specialized LLM-driven experts that generate EvidenceItems.
    """
    
    def __init__(self, agent_id: str, llm_client: Any):
        """
        Initializes the L1 agent.
        
        Args:
            agent_id (str): The unique identifier for the agent (from registry.yaml).
            llm_client (Any): An instance of an LLM client (e.g., GeminiPoolManager).
        """
        self.agent_id = agent_id
        self.llm_client = llm_client

    @abstractmethod
    def run(self, state: PipelineState, dependencies: Dict[str, 'EvidenceItem']) -> 'EvidenceItem':
        """
        The main execution method for the agent.
        
        It takes the current pipeline state and any dependency outputs,
        performs its specialized analysis, and returns a single EvidenceItem.
        
        Args:
            state (PipelineState): The current state of the analysis pipeline.
            dependencies (Dict[str, 'EvidenceItem']): Outputs from any upstream tasks this agent depends on.
            
        Returns:
            EvidenceItem: An object (defined in core.schemas) containing the
                          agent's findings, confidence, and supporting data.
        """
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id='{self.agent_id}')>"

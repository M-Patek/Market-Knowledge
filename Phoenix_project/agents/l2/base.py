"""
Base class for all L2 (Metacognition & Arbitration) Agents.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.core.schemas.evidence_schema import EvidenceItem

class BaseL2Agent(ABC):
    """
    Abstract Base Class for all L2 agents.
    L2 agents are responsible for integrating, cross-validating, and
    making advanced decisions based on the EvidenceItems from L1.
    """
    
    def __init__(self, agent_id: str, llm_client: Any = None):
        """
        Initializes the L2 agent.
        
        Args:
            agent_id (str): The unique identifier for the agent (from registry.yaml).
            llm_client (Any, optional): An instance of an LLM client. Required by
                                      agents like critic, adversary, and fusion.
        """
        self.agent_id = agent_id
        self.llm_client = llm_client

    @abstractmethod
    def run(self, state: PipelineState, evidence_items: List[EvidenceItem]) -> Any:
        """
        The main execution method for the agent.
        
        It takes the current pipeline state and the list of L1 EvidenceItems,
        performs its specialized analysis (e.g., criticism, fusion), and
        returns its result. The return type varies by agent (e.g., CriticResult, FusionResult).
        
        Args:
            state (PipelineState): The current state of the analysis pipeline.
            evidence_items (List[EvidenceItem]): The collected list of outputs from the L1 agents.
            
        Returns:
            Any: The specific result object for that agent's task.
        """
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id='{self.agent_id}')>"

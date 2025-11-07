"""
Base class for all L2 (Metacognition & Arbitration) Agents.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List # 确保 Dict 和 Any 被导入

from Phoenix_project.core.pipeline_state import PipelineState
# from Phoenix_project.core.schemas.evidence_schema import EvidenceItem # 不再需要

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
    def run(self, state: PipelineState, dependencies: Dict[str, Any]) -> Any:
        """
        The main execution method for the agent.
        
        It takes the current pipeline state and a dictionary of dependency outputs,
        performs its specialized analysis, and returns its result.
        
        Args:
            state (PipelineState): The current state of the analysis pipeline.
            dependencies (Dict[str, Any]): A dictionary mapping dependency agent IDs
                                         to their execution results.
            
        Returns:
            Any: The specific result object for that agent's task.
        """
        pass

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id='{self.agent_id}')>"

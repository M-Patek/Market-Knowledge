"""
Base class for all L1 (Expert) Agents.
[Beta FIX] Added safe_run for error handling boundary.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List
import logging

# We use a forward reference 'EvidenceItem' to avoid circular dependencies
# or needing to import the schema file just for this base class definition.
# The actual EvidenceItem schema will be defined in core/schemas/.
from Phoenix_project.core.pipeline_state import PipelineState

logger = logging.getLogger(__name__)

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
    def run(self, state: PipelineState, dependencies: Dict[str, Any]) -> Any:
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

    def safe_run(self, state: PipelineState, dependencies: Dict[str, Any]) -> Any:
        """
        [Beta FIX] A defensive wrapper around the abstract run method.
        Handles exceptions and returns a structured error result if the agent crashes.
        Prevents a single agent failure from bringing down the entire pipeline.
        """
        try:
            logger.info(f"Agent {self.agent_id} starting execution.")
            # 可以在这里添加统一的输入验证逻辑
            if not dependencies and state is None:
                 logger.warning(f"Agent {self.agent_id} received empty state and dependencies.")

            result = self.run(state, dependencies)
            return result

        except Exception as e:
            logger.error(f"Agent {self.agent_id} CRASHED during execution: {e}", exc_info=True)
            
            # 返回一个标准的错误 EvidenceItem (模拟结构，需根据实际 Schema 调整)
            # 这里假设返回一个包含错误信息的对象或字典，确保 Executor 能处理
            return {
                "source": self.agent_id,
                "content": "N/A",
                "status": "ERROR",
                "error_message": str(e),
                "confidence": 0.0
            }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(id='{self.agent_id}')>"

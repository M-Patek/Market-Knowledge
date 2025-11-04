"""
L2 Agent: Critic
Refactored from evaluation/critic.py.
Responsible for "Criticism & Fact Check" on L1 EvidenceItems.
"""
from typing import Any, List

from Phoenix_project.agents.l2.base import BaseL2Agent
from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.core.schemas.evidence_schema import EvidenceItem
from Phoenix_project.core.schemas.critic_result import CriticResult

class CriticAgent(BaseL2Agent):
    """
    Implements the L2 Critic agent.
    Inherits from BaseL2Agent and implements the run method
    to critique L1 EvidenceItems.
    """
    
    def run(self, state: PipelineState, evidence_items: List[EvidenceItem]) -> List[CriticResult]:
        """
        Reviews L1 Agent outputs for logical consistency, factual accuracy,
        and constraint compliance.
        
        Args:
            state (PipelineState): The current state of the analysis pipeline.
            evidence_items (List[EvidenceItem]): The collected list of outputs
                                                 from the L1 agents.
            
        Returns:
            List[CriticResult]: A list of critique objects, one for each
                                piece of evidence.
        """
        critiques = []
        for item in evidence_items:
            # Format the evidence for critique
            text_to_critique = f"Agent '{item.agent_id}' produced the following evidence with confidence {item.confidence}:\n{item.content}"
            
            # TODO: Implement actual LLM call using self.llm_client.
            # This logic should be adapted from the original Critic class:
            # 1. Get a prompt from a prompt_manager.
            # 2. Call self.llm_client.send_request(prompt=..., model=...).
            # 3. Parse the JSON response.
            
            # This is a mock critique result.
            if "strong" in item.content:
                mock_critique = "Critique: The term 'strong' is vague. Analysis lacks specific supporting data."
                mock_is_valid = False
                mock_confidence_adj = 0.8
                mock_flags = ["VAGUE_LANGUAGE", "DATA_MISSING"]
            else:
                mock_critique = "Critique: Analysis appears logical and consistent with provided context."
                mock_is_valid = True
                mock_confidence_adj = 1.0
                mock_flags = []

            critiques.append(CriticResult(
                agent_id=self.agent_id,
                target_evidence_id=item.id,
                is_valid=mock_is_valid,
                critique=mock_critique,
                confidence_adjustment=mock_confidence_adj,
                flags=mock_flags
            ))
            
        return critiques

    def __repr__(self) -> str:
        return f"<CriticAgent(id='{self.agent_id}')>"

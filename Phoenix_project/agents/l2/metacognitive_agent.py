"""
L2 Agent: Metacognitive Agent
Refactored from ai/metacognitive_agent.py.
Responsible for "Supervision" of L1/L2 agent reasoning (CoT).
"""
from typing import Any, List

from agents.l2.base import BaseL2Agent
from core.pipeline_state import PipelineState
from core.schemas.evidence_schema import EvidenceItem
from core.schemas.supervision_result import SupervisionResult

class MetacognitiveAgent(BaseL2Agent):
    """
    Implements the L2 Metacognitive agent.
    This agent monitors the CoT of other agents to identify
    divergence and potential hallucinations.
    """
    
    def run(self, state: PipelineState, evidence_items: List[EvidenceItem]) -> SupervisionResult:
        """
        Monitors the CoT of L1 and L2 Agents behind the scenes.
        
        Args:
            state (PipelineState): The current state, used to access CoT traces.
            evidence_items (List[EvidenceItem]): The collected list of L1 outputs.
            
        Returns:
            SupervisionResult: A single result object summarizing the findings.
        """
        
        # TODO: Implement actual LLM call using self.llm_client.
        # This logic should be adapted from the original MetacognitiveAgent:
        # 1. Access CoT traces from the PipelineState (e.g., state.get_all_cot_traces())
        # 2. Serialize the traces for a prompt.
        # 3. Ask the LLM to identify 'hallucinations', 'divergence', 'bias'.
        # 4. Parse the response into the SupervisionResult.
        
        # This is a mock supervision result.
        # It simulates analyzing the L1 agents' reasoning.
        
        target_agents = [item.agent_id for item in evidence_items]
        mock_summary = f"Supervision complete for {len(target_agents)} L1 agents. No hallucinations detected. Minor divergence in confidence levels observed."

        return SupervisionResult(
            agent_id=self.agent_id,
            analysis_summary=mock_summary,
            target_agent_ids=target_agents,
            flags=["MINOR_DIVERGENCE"]
        )

    def __repr__(self) -> str:
        return f"<MetacognitiveAgent(id='{self.agent_id}')>"

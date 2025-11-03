"""
L2 Agent: Adversary
Refactored from ai/counterfactual_tester.py.
Responsible for "Counter-Argument" and "Pressure Testing" L1 Evidence.
"""
from typing import Any, List

from agents.l2.base import BaseL2Agent
from core.pipeline_state import PipelineState
from core.schemas.evidence_schema import EvidenceItem, EvidenceType
from core.schemas.adversary_result import AdversaryResult

class AdversaryAgent(BaseL2Agent):
    """
    Implements the L2 Adversary agent.
    Inherits from BaseL2Agent and implements the run method
    to generate counter-arguments against L1 EvidenceItems.
    """
    
    def run(self, state: PipelineState, evidence_items: List[EvidenceItem]) -> List[AdversaryResult]:
        """
        Actively seeks flaws and counter-arguments against the EvidenceItems
        provided by L1 Agents.
        
        Args:
            state (PipelineState): The current state of the analysis pipeline.
            evidence_items (List[EvidenceItem]): The collected list of outputs
                                                 from the L1 agents.
            
        Returns:
            List[AdversaryResult]: A list of counter-argument objects.
        """
        counter_arguments = []
        for item in evidence_items:
            # Format the evidence for pressure testing
            text_to_test = f"Agent '{item.agent_id}' claimed: '{item.content}'"
            
            # TODO: Implement actual LLM call using self.llm_client.
            # This logic should be adapted from the original CounterfactualTester:
            # 1. Get a prompt from a prompt_manager to generate a counter-argument.
            # 2. Call self.llm_client.send_request(prompt=...).
            # 3. Parse the JSON response.
            
            # This is a mock counter-argument.
            if item.evidence_type == EvidenceType.FUNDAMENTAL:
                mock_counter = "Counter: What if the 'strong earnings growth' is due to a one-time asset sale, not core business improvement?"
                mock_success = True
                mock_impact = -0.3
            else:
                mock_counter = "Counter: The 'bullish MACD crossover' could be a false signal, as volume on the breakout was below average."
                mock_success = True
                mock_impact = -0.25

            counter_arguments.append(AdversaryResult(
                agent_id=self.agent_id,
                target_evidence_id=item.id,
                counter_argument=mock_counter,
                is_challenge_successful=mock_success,
                confidence_impact=mock_impact
            ))
            
        return counter_arguments

    def __repr__(self) -> str:
        return f"<AdversaryAgent(id='{self.agent_id}')>"

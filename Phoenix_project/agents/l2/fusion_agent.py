"""
L2 Agent: Fusion
Refactored from fusion/synthesizer.py.
Responsible for "Fusion & Arbitration" of all L1 and L2 outputs.
"""
from typing import Any, List, Dict
from decimal import Decimal

from agents.l2.base import BaseL2Agent
from core.pipeline_state import PipelineState
from core.schemas.evidence_schema import EvidenceItem
from core.schemas.critic_result import CriticResult
from core.schemas.adversary_result import AdversaryResult
from core.schemas.fusion_result import FusionResult

class FusionAgent(BaseL2Agent):
    """
    Implements the L2 Fusion agent.
    Inherits from BaseL2Agent and implements the run method
    to synthesize all L1/L2 data into a final L2 decision.
    """
    
    def run(self, state: PipelineState, evidence_items: List[EvidenceItem]) -> FusionResult:
        """
        Synthesizes all evidence (L1) and critiques (L2) to form a
        unified, high-confidence preliminary decision (FusionResult).
        
        Args:
            state (PipelineState): The current state, used to retrieve L2 results.
            evidence_items (List[EvidenceItem]): The collected list of L1 outputs.
            
        Returns:
            FusionResult: The single, unified decision for the L3 layer.
        """
        
        # Retrieve the outputs from the other L2 agents from the state
        # (This assumes the orchestrator ran them and stored their results)
        critic_results: List[CriticResult] = state.get_results_by_type(CriticResult)
        adversary_results: List[AdversaryResult] = state.get_results_by_type(AdversaryResult)

        # We assume the target symbol is accessible via the state.
        task_query_data = state.get_main_task_query() 
        target_symbol = task_query_data.get("symbol", "UNKNOWN")

        # TODO: Implement actual LLM call using self.llm_client.
        # This logic should be adapted from the original Synthesizer/Arbitrator:
        # 1. Format a complex prompt with:
        #    - All L1 EvidenceItems (content, confidence)
        #    - All L2 CriticResults (critiques, flags)
        #    - All L2 AdversaryResults (counter-arguments)
        # 2. Ask the LLM to act as an "Arbitrator" to produce a final
        #    decision, confidence, and summary.
        # 3. Parse the response into the FusionResult schema.
        
        # This is a mock fusion result.
        # We create a summary that references the (mock) inputs.
        critic_summary = f"{len(critic_results)} critiques" if critic_results else "no critiques"
        adversary_summary = f"{len(adversary_results)} counter-arguments" if adversary_results else "no counter-arguments"
        
        mock_summary = f"Synthesized decision for {target_symbol}: Based on {len(evidence_items)} evidence items, {critic_summary}, and {adversary_summary}. After arbitration, the consensus is a cautious BUY."
        
        return FusionResult(
            target_symbol=target_symbol,
            decision="BUY",
            confidence=0.65,
            reasoning=mock_summary,
            uncertainty=0.35,
            supporting_evidence_ids=[item.id for item in evidence_items],
            conflicting_evidence_ids=[], # This would be populated by the LLM
            metadata={
                "critic_results_count": len(critic_results),
                "adversary_results_count": len(adversary_results)
            }
        )

    def __repr__(self) -> str:
        return f"<FusionAgent(id='{self.agent_id}')>"

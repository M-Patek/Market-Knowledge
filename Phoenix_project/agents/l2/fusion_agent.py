"""
L2 Agent: Fusion & Synthesis
"""
from typing import Any, Dict, List
from decimal import Decimal

from Phoenix_project.agents.l2.base import BaseL2Agent
from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.core.schemas.evidence_schema import EvidenceItem
from Phoenix_project.core.schemas.fusion_result import FusionResult

class FusionAgent(BaseL2Agent):
    """
    Implements the L2 Fusion agent.
    This agent is responsible for synthesizing all L1 EvidenceItems
    into a single, unified FusionResult.
    
    In a real system, this would be a complex LLM call.
    """
    
    def run(self, state: PipelineState, evidence_items: List[EvidenceItem]) -> FusionResult:
        """
        Synthesizes L1 evidence into a final decision.
        
        Args:
            state (PipelineState): The current state of the analysis pipeline.
            evidence_items (List[EvidenceItem]): The collected list of outputs from the L1 agents.
            
        Returns:
            FusionResult: A unified decision object.
        """
        
        # This is a mock synthesis logic.
        # A real implementation would:
        # 1. Get the main query (e.g., "Analyze NVDA") from the state.
        # 2. Serialize all 'evidence_items' into a context block.
        # 3. Use an LLM prompt (like prompts/fusion.json) to synthesize them.
        
        task_query_data = state.get_main_task_query()
        target_symbol = task_query_data.get("symbol", "UNKNOWN")

        if not evidence_items:
            return FusionResult(
                final_decision="HOLD",
                reasoning="No L1 evidence was generated to make a decision.",
                confidence=Decimal("0.50"),
                metadata={"synthesis_model": "MockFusionAgent"}
            )
            
        # Mock logic: If any L1 agent had high confidence, we'll go with that.
        # (This is overly simple, but demonstrates the flow)
        
        best_evidence = max(evidence_items, key=lambda x: x.confidence)
        
        # A simple mock decision based on the "best" evidence type
        mock_decision = "HOLD"
        if best_evidence.evidence_type == EvidenceType.CATALYST and best_evidence.confidence > 0.8:
            mock_decision = "BUY"
        elif best_evidence.evidence_type == EvidenceType.FUNDAMENTAL and best_evidence.confidence < 0.3:
            mock_decision = "SELL"
            
        synthesized_reasoning = f"Mock Synthesis for {target_symbol}:\n"
        synthesized_reasoning += f"Decision based on strongest L1 evidence (Agent: {best_evidence.agent_id}, Type: {best_evidence.evidence_type.value}).\n"
        synthesized_reasoning += f"Evidence Content: {best_evidence.content}\n"
        synthesized_reasoning += f"Final Decision: {mock_decision} with confidence {best_evidence.confidence}."

        return FusionResult(
            final_decision=mock_decision,
            reasoning=synthesized_reasoning,
            confidence=best_evidence.confidence,
            supporting_evidence_ids=[item.item_id for item in evidence_items],
            metadata={"synthesis_model": "MockFusionAgent", "strongest_agent": best_evidence.agent_id}
        )

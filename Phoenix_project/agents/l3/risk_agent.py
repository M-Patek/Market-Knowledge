"""
L3 Agent: Risk Agent
Refactored from training/drl/agents/risk_agent.py.
Responsible for "Capital Adjustment."
"""
from typing import Any

from Phoenix_project.agents.l3.base import BaseL3Agent
from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.core.schemas.fusion_result import FusionResult
from Phoenix_project.core.schemas.risk_schema import RiskAdjustment

class RiskAgent(BaseL3Agent):
    """
    Implements the L3 Risk agent.
    Inherits from BaseL3Agent and implements the run method
    to convert an L2 FusionResult into a capital adjustment.
    """
    
    def run(self, state: PipelineState, fusion_result: FusionResult) -> RiskAdjustment:
        """
        Assesses trade risk and determines the capital allocation ratio
        (capital_modifier) for the signal.
        
        Args:
            state (PipelineState): The current state of the analysis pipeline.
            fusion_result (FusionResult): The unified decision output from the L2 layer.
            
        Returns:
            RiskAdjustment: A standardized object containing the capital modifier.
        """
        
        # TODO: Implement actual DRL/Quant model logic.
        # This logic would use self.model_client to process the
        # fusion_result's uncertainty, confidence, and market volatility (from state).
        
        # This is a mock risk adjustment.
        # We base the modifier on the L2 uncertainty score.
        uncertainty = fusion_result.uncertainty
        
        # Simple mock logic: modifier = 1.0 - uncertainty
        capital_modifier = max(0.0, 1.0 - uncertainty)
        reasoning = f"Adjusting capital based on L2 uncertainty score of {uncertainty:.2f}."

        return RiskAdjustment(
            agent_id=self.agent_id,
            target_symbol=fusion_result.target_symbol,
            capital_modifier=capital_modifier,
            reasoning=reasoning,
            metadata={"source_uncertainty": uncertainty}
        )

    def __repr__(self) -> str:
        return f"<RiskAgent(id='{self.agent_id}')>"

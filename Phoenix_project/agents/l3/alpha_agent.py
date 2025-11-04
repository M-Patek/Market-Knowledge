"""
L3 Agent: Alpha Agent
Refactored from training/drl/agents/alpha_agent.py.
Responsible for "Signal Generation."
"""
from typing import Any

from Phoenix_project.agents.l3.base import BaseL3Agent
from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.core.schemas.fusion_result import FusionResult
from Phoenix_project.core.schemas.data_schema import Signal

class AlphaAgent(BaseL3Agent):
    """
    Implements the L3 Alpha agent.
    Inherits from BaseL3Agent and implements the run method
    to convert an L2 FusionResult into a trading Signal.
    """
    
    def run(self, state: PipelineState, fusion_result: FusionResult) -> Signal:
        """
        Converts the preliminary L2 decision (FusionResult) into a
        refined trading signal (Signal).
        
        Args:
            state (PipelineState): The current state of the analysis pipeline.
            fusion_result (FusionResult): The unified decision output from the L2 layer.
            
        Returns:
            Signal: A standardized Signal object for the execution layer.
        """
        
        # TODO: Implement actual DRL/Quant model logic.
        # This logic would use self.model_client (the loaded DRL model)
        # to process the fusion_result.
        
        # For now, we translate the L2 decision directly into the Signal schema.
        # This is the final step in our L1->L2->L3 data flow.

        return Signal(
            symbol=fusion_result.target_symbol,
            signal_type=fusion_result.decision.upper(), # e.g., "BUY", "SELL"
            strength=fusion_result.confidence,
            metadata={"source_agent": self.agent_id, "fusion_id": fusion_result.id}
        )

    def __repr__(self) -> str:
        return f"<AlphaAgent(id='{self.agent_id}')>"

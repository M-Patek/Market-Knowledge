"""
L1 Agent: Macro Strategist
"""
from typing import Any, Dict
from decimal import Decimal

from agents.l1.base import BaseL1Agent
from core.pipeline_state import PipelineState
from core.schemas.evidence_schema import EvidenceItem, EvidenceType

class MacroStrategistAgent(BaseL1Agent):
    """
    Implements the L1 Macro Strategist agent.
    Inherits from BaseL1Agent and implements the run method.
    """
    
    def run(self, state: PipelineState, dependencies: Dict[str, Any]) -> EvidenceItem:
        """
        Analyzes macroeconomic indicators and central bank policies.
        
        Args:
            state (PipelineState): The current state of the analysis pipeline.
            dependencies (Dict[str, Any]): Outputs from any upstream tasks.
            
        Returns:
            EvidenceItem: An EvidenceItem object with the macro analysis.
        """
        
        # This agent may not be symbol-specific, but context-specific.
        # For now, we'll link it to the target symbol's market (e.g., 'US').
        task_query_data = state.get_main_task_query() 
        target_symbol = task_query_data.get("symbol", "UNKNOWN")
        
        # TODO: Implement actual LLM call to analyze macro data (e.g., CPI, Fed policy).
        # This is a mock analysis result.
        mock_analysis_content = f"MACRO: Federal Reserve indicates a hawkish stance, potentially tightening liquidity. This may act as a headwind for equities like {target_symbol}."

        return EvidenceItem(
            agent_id=self.agent_id,
            symbols=[target_symbol, "SPY"], # Macro affects the whole market
            evidence_type=EvidenceType.MACRO,
            content=mock_analysis_content,
            confidence=Decimal("0.70"),
            data_horizon="Medium-Term (Monthly/Quarterly)",
            metadata={"source": "Mock Fed Watch"}
        )

    def __repr__(self) -> str:
        return f"<MacroStrategistAgent(id='{self.agent_id}')>"

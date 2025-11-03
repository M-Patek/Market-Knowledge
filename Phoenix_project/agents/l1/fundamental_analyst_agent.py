"""
L1 Agent: Fundamental Analyst
"""
from typing import Any, Dict
from decimal import Decimal

from agents.l1.base import BaseL1Agent
from core.pipeline_state import PipelineState
from core.schemas.evidence_schema import EvidenceItem, EvidenceType

class FundamentalAnalystAgent(BaseL1Agent):
    """
    Implements the L1 Fundamental Analyst agent.
    Inherits from BaseL1Agent and implements the run method.
    """
    
    def run(self, state: PipelineState, dependencies: Dict[str, Any]) -> EvidenceItem:
        """
        Performs fundamental analysis based on the pipeline state.
        
        Args:
            state (PipelineState): The current state of the analysis pipeline.
            dependencies (Dict[str, Any]): Outputs from any upstream tasks.
            
        Returns:
            EvidenceItem: An EvidenceItem object with the fundamental analysis.
        """
        
        # We assume the target symbol is accessible via the state.
        # This 'get_main_task_query' is a placeholder for state.main_task or similar.
        task_query_data = state.get_main_task_query() 
        target_symbol = task_query_data.get("symbol", "UNKNOWN")
        
        # TODO: Implement actual LLM call to perform fundamental analysis.
        # This is a mock analysis result.
        mock_analysis_content = f"Fundamental analysis for {target_symbol} shows strong earnings growth and a healthy balance sheet. Q4 revenue exceeded expectations."

        return EvidenceItem(
            agent_id=self.agent_id,
            symbols=[target_symbol],
            evidence_type=EvidenceType.FUNDAMENTAL,
            content=mock_analysis_content,
            confidence=Decimal("0.85"),
            data_horizon="Medium to Long-Term (Quarterly/Annually)",
            metadata={"source": "Mock Analysis"}
        )

    def __repr__(self) -> str:
        return f"<FundamentalAnalystAgent(id='{self.agent_id}')>"

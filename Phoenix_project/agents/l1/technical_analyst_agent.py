"""
L1 Agent: Technical Analyst
"""
from typing import Any, Dict
from decimal import Decimal

from Phoenix_project.agents.l1.base import BaseL1Agent
from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.core.schemas.evidence_schema import EvidenceItem, EvidenceType

class TechnicalAnalystAgent(BaseL1Agent):
    """
    Implements the L1 Technical Analyst agent.
    Inherits from BaseL1Agent and implements the run method.
    """
    
    def run(self, state: PipelineState, dependencies: Dict[str, Any]) -> EvidenceItem:
        """
        Performs technical analysis based on the pipeline state.
        
        Args:
            state (PipelineState): The current state of the analysis pipeline.
            dependencies (Dict[str, Any]): Outputs from any upstream tasks.
            
        Returns:
            EvidenceItem: An EvidenceItem object with the technical analysis.
        """
        
        # We assume the target symbol is accessible via the state.
        task_query_data = state.get_main_task_query() 
        target_symbol = task_query_data.get("symbol", "UNKNOWN")
        
        # TODO: Implement actual LLM call to perform technical analysis.
        # This is a mock analysis result.
        mock_analysis_content = f"Technical analysis for {target_symbol} shows a bullish MACD crossover and a breakout above the 50-day moving average. Resistance at $150."

        return EvidenceItem(
            agent_id=self.agent_id,
            symbols=[target_symbol],
            evidence_type=EvidenceType.TECHNICAL,
            content=mock_analysis_content,
            confidence=Decimal("0.75"),
            data_horizon="Short-Term/Ultra Short-Term (High-Frequency)",
            metadata={"source": "Mock Analysis"}
        )

    def __repr__(self) -> str:
        return f"<TechnicalAnalystAgent(id='{self.agent_id}')>"

"""
L1 Agent: Geopolitical & Regulatory Analyst
"""
from typing import Any, Dict
from decimal import Decimal

from Phoenix_project.agents.l1.base import BaseL1Agent
from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.core.schemas.evidence_schema import EvidenceItem, EvidenceType

class GeopoliticalAnalystAgent(BaseL1Agent):
    """
    Implements the L1 Geopolitical & Regulatory Analyst agent.
    Inherits from BaseL1Agent and implements the run method.
    """
    
    def run(self, state: PipelineState, dependencies: Dict[str, Any]) -> EvidenceItem:
        """
        Assesses geopolitical risks and changes in industry regulatory policy.
        
        Args:
            state (PipelineState): The current state of the analysis pipeline.
            dependencies (Dict[str, Any]): Outputs from any upstream tasks.
            
        Returns:
            EvidenceItem: An EvidenceItem object with the geopolitical analysis.
        """
        
        # We assume the target symbol is accessible via the state.
        task_query_data = state.get_main_task_query() 
        target_symbol = task_query_data.get("symbol", "UNKNOWN")
        
        # TODO: Implement actual LLM call to scan for geopolitical/regulatory news.
        # This is a mock analysis result.
        mock_analysis_content = f"GEOPOLITICAL: New tariffs on semiconductor imports announced, which could impact {target_symbol}'s supply chain costs."

        return EvidenceItem(
            agent_id=self.agent_id,
            symbols=[target_symbol],
            evidence_type=EvidenceType.GEOPOLITICAL,
            content=mock_analysis_content,
            confidence=Decimal("0.80"),
            data_horizon="Long-Term (Trend-Based)",
            metadata={"source": "Mock Regulatory Feed"}
        )

    def __repr__(self) -> str:
        return f"<GeopoliticalAnalystAgent(id='{self.agent_id}')>"

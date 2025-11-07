"""
L2 Agent: Adversary
Refactored from ai/counterfactual_tester.py.
Responsible for "Counter-Argument" and "Pressure Testing" L1 Evidence.
"""
from typing import Any, List, Dict

from Phoenix_project.agents.l2.base import BaseL2Agent
from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.core.schemas.evidence_schema import EvidenceItem, EvidenceType
from Phoenix_project.core.schemas.adversary_result import AdversaryResult

class AdversaryAgent(BaseL2Agent):
    """
    Implements the L2 Adversary agent.
    Inherits from BaseL2Agent and implements the run method
    to generate counter-arguments against L1 EvidenceItems.
    """
    
    # 签名已更新：接受 dependencies 而不是 evidence_items
    def run(self, state: PipelineState, dependencies: Dict[str, Any]) -> List[AdversaryResult]:
        """
        Actively seeks flaws and counter-arguments against the EvidenceItems
        provided by L1 Agents.
        
        Args:
            state (PipelineState): The current state of the analysis pipeline.
            dependencies (Dict[str, Any]): The dictionary of outputs from dependent tasks
                                         (expected to contain L1 EvidenceItems).
            
        Returns:
            List[AdversaryResult]: A list of counter-argument objects.
        """
        
        # --- 新增逻辑：从 dependencies 提取 evidence_items ---
        evidence_items: List[EvidenceItem] = []
        for result in dependencies.values():
            if isinstance(result, EvidenceItem):
                evidence_items.append(result)
            elif isinstance(result, list): # 处理 L1 agent 可能返回列表的情况
                for item in result:
                    if isinstance(item, EvidenceItem):
                        evidence_items.append(item)
        # --- 新增逻辑结束 ---
        
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

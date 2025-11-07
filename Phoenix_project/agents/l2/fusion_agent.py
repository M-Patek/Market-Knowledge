"""
L2 Agent: Fusion & Synthesis
"""
from typing import Any, Dict, List
from decimal import Decimal

from Phoenix_project.agents.l2.base import BaseL2Agent
from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.core.schemas.evidence_schema import EvidenceItem, EvidenceType
from Phoenix_project.core.schemas.fusion_result import FusionResult

class FusionAgent(BaseL2Agent):
    """
    Implements the L2 Fusion agent.
    This agent is responsible for synthesizing all L1 EvidenceItems
    into a single, unified FusionResult.
    
    In a real system, this would be a complex LLM call.
    """
    
    # 签名已更新，以匹配 Executor
    def run(self, state: PipelineState, dependencies: Dict[str, Any]) -> FusionResult:
        """
        Synthesizes L1 evidence into a final decision.
        
        Args:
            state (PipelineState): The current state of the analysis pipeline.
            dependencies (Dict[str, Any]): Outputs from L1 agents. 
                                         Values are expected to be EvidenceItem.
            
        Returns:
            FusionResult: A unified decision object.
        """
        
        # --- 新增逻辑：从 dependencies 提取 evidence_items ---
        # 假设 L1 依赖项都返回 EvidenceItem
        evidence_items: List[EvidenceItem] = []
        for result in dependencies.values():
            if isinstance(result, EvidenceItem):
                evidence_items.append(result)
            elif isinstance(result, list): # 处理 L1 agent 可能返回列表的情况
                for item in result:
                    if isinstance(item, EvidenceItem):
                        evidence_items.append(item)
        # --- 新增逻辑结束 ---
        

        # vvvv 现有的核心逻辑保持不变 vvvv
        
        task_query_data = state.get_main_task_query()
        target_symbol = task_query_data.get("symbol", "UNKNOWN")

        if not evidence_items:
            # [FIXED] Added missing required fields: target_symbol, uncertainty
            # Ensured confidence is float
            return FusionResult(
                target_symbol=target_symbol,
                final_decision="HOLD",
                reasoning="No L1 evidence was generated to make a decision.",
                confidence=0.50,
                uncertainty=0.50, # Mock uncertainty
                metadata={"synthesis_model": "MockFusionAgent"},
                contributing_agents=[]  # <-- 1. 遵照主人指示：使用空列表以匹配类型
            )
            
        best_evidence = max(evidence_items, key=lambda x: x.confidence)
        
        mock_decision = "HOLD"
        if best_evidence.evidence_type == EvidenceType.CATALYST and best_evidence.confidence > 0.8:
            mock_decision = "BUY"
        elif best_evidence.evidence_type == EvidenceType.FUNDAMENTAL and best_evidence.confidence < 0.3:
            mock_decision = "SELL"
            
        synthesized_reasoning = f"Mock Synthesis for {target_symbol}:\n"
        synthesized_reasoning += f"Decision based on strongest L1 evidence (Agent: {best_evidence.agent_id}, Type: {best_evidence.evidence_type.value}).\n"
        synthesized_reasoning += f"Evidence Content: {best_evidence.content}\n"
        synthesized_reasoning += f"Final Decision: {mock_decision} with confidence {best_evidence.confidence}."

        # [FIXED] Added missing fields: target_symbol, uncertainty
        # [FIXED] Corrected item.item_id to item.id
        # Ensured confidence is float
        return FusionResult(
            target_symbol=target_symbol,
            final_decision=mock_decision,
            reasoning=synthesized_reasoning,
            confidence=float(best_evidence.confidence),
            uncertainty=float(1.0 - best_evidence.confidence), # Mock uncertainty based on confidence
            supporting_evidence_ids=[item.id for item in evidence_items],
            metadata={"synthesis_model": "MockFusionAgent", "strongest_agent": best_evidence.agent_id},
            contributing_agents=[]  # <-- 2. 遵照主人指示：使用空列表以匹配类型
        )

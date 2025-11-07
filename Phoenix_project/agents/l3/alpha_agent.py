"""
L3 Agent: Alpha Agent
Refactored from training/drl/agents/alpha_agent.py.
Responsible for "Signal Generation."
"""
from typing import Any, Dict # 确保 Dict 被导入

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
    
    # 签名已更新，以匹配 Executor
    def run(self, state: PipelineState, dependencies: Dict[str, Any]) -> Signal:
        """
        Converts the preliminary L2 decision (FusionResult) into a
        refined trading signal (Signal).
        
        Args:
            state (PipelineState): The current state of the analysis pipeline.
            dependencies (Dict[str, Any]): Outputs from L2 agents. 
                                         Should contain one FusionResult.
            
        Returns:
            Signal: A standardized Signal object for the execution layer.
        """
        
        # --- 新增逻辑：从 dependencies 提取 fusion_result ---
        fusion_result: FusionResult = None
        for result in dependencies.values():
            if isinstance(result, FusionResult):
                fusion_result = result
                break  # 假设只有一个 FusionResult 依赖项

        if fusion_result is None:
            raise ValueError(f"Agent {self.agent_id} did not find required FusionResult in dependencies.")
        # --- 新增逻辑结束 ---
        

        # vvvv 现有的核心逻辑保持不变 vvvv
        # (注意：此处的 .target_symbol, .decision, 和 .id
        # 与 l2/fusion_agent.py 的输出不匹配，
        # 但按照“不修改逻辑”的指示，此处保持原样)
        
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

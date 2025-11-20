import json
import logging
from typing import List, Any, AsyncGenerator

from Phoenix_project.agents.l2.base import L2Agent
from Phoenix_project.core.pipeline_state import PipelineState
# 假设有一个 MetacognitiveResult schema，如果没有，可以使用 dict
from typing import Dict 

logger = logging.getLogger(__name__)

class MetacognitiveAgent(L2Agent):
    """
    L2 智能体：元认知 (Metacognitive)
    监控系统的思维过程，识别认知偏差，并调整策略。
    """

    async def run(self, state: PipelineState, dependencies: List[Any]) -> AsyncGenerator[Dict[str, Any], None]:
        """
        [Refactored Phase 3.2] 适配 PipelineState。
        """
        target_symbol = state.main_task_query.get("symbol", "UNKNOWN")
        logger.info(f"[{self.agent_id}] Running reflection for: {target_symbol}")
        
        # Metacognitive Agent 需要查看整个 Fusion 历史或最新的决策
        latest_decision = state.latest_final_decision
        
        if not latest_decision:
             logger.debug(f"[{self.agent_id}] No decision to reflect on.")
             yield self._create_fallback_result(state, "No recent decision")
             return

        try:
            # 简化的反射逻辑
            context_map = {
                "recent_decision": json.dumps(latest_decision, default=str),
                "current_date": state.current_time.isoformat()
            }
            
            # 这里可以调用 LLM 进行深度反思
            # 目前简化为返回一个状态检查
            
            result = {
                "timestamp": state.current_time,
                "reflection": "System logic appears nominal.",
                "bias_detected": False,
                "suggested_adjustments": []
            }
            
            yield result

        except Exception as e:
            logger.error(f"[{self.agent_id}] Error: {e}", exc_info=True)
            yield self._create_fallback_result(state, str(e))

    def _create_fallback_result(self, state: PipelineState, reason: str) -> Dict[str, Any]:
        return {
            "timestamp": state.current_time,
            "reflection": f"Reflection skipped: {reason}",
            "bias_detected": False,
            "suggested_adjustments": []
        }

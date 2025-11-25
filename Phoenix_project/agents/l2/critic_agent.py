import json
import logging
from typing import List, Any, AsyncGenerator

from Phoenix_project.agents.l2.base import L2Agent
from Phoenix_project.core.schemas.evidence_schema import EvidenceItem
from Phoenix_project.core.schemas.critic_result import CriticResult
from Phoenix_project.core.pipeline_state import PipelineState
from pydantic import ValidationError

logger = logging.getLogger(__name__)

class CriticAgent(L2Agent):
    """
    L2 智能体：批评者 (Critic)
    评估证据的质量、逻辑一致性和潜在偏见。
    """

    async def run(self, state: PipelineState, dependencies: List[Any]) -> AsyncGenerator[CriticResult, None]:
        """
        [Refactored Phase 3.2] 适配 PipelineState，增加鲁棒性和兜底机制。
        """
        target_symbol = state.main_task_query.get("symbol", "UNKNOWN")
        logger.info(f"[{self.agent_id}] Running Critique for: {target_symbol}")

        # 1. 提取证据
        evidence_items = []
        if dependencies:
            for item in dependencies:
                if isinstance(item, EvidenceItem):
                    evidence_items.append(item)
                elif isinstance(item, dict) and "content" in item:
                    try:
                        evidence_items.append(EvidenceItem(**item))
                    except Exception as e:
                        logger.warning(f"[{self.agent_id}] Skipping invalid evidence item: {e}")

        if not evidence_items:
            logger.warning(f"[{self.agent_id}] No evidence to critique. Yielding fallback.")
            yield self._create_fallback_result(state, target_symbol, "No evidence provided.")
            return

        agent_prompt_name = "l2_critic"

        try:
            # 2. 准备 Prompt
            evidence_json_list = self._safe_prepare_context([item.model_dump() for item in evidence_items])
            
            context_map = {
                "target_symbol": target_symbol,
                "evidence_json_list": evidence_json_list
            }

            # 3. 调用 LLM
            response_str = await self.llm_client.run_llm_task(
                agent_prompt_name=agent_prompt_name,
                context_map=context_map
            )

            if not response_str:
                yield self._create_fallback_result(state, target_symbol, "LLM returned empty response.")
                return

            # 4. 解析结果
            response_data = json.loads(response_str)
            
            if "timestamp" not in response_data:
                response_data["timestamp"] = state.current_time

            result = CriticResult.model_validate(response_data)
            object.__setattr__(result, 'timestamp', state.current_time)

            logger.info(f"[{self.agent_id}] Critique complete. Valid: {result.is_valid}")
            yield result

        except Exception as e:
            logger.error(f"[{self.agent_id}] Error: {e}", exc_info=True)
            yield self._create_fallback_result(state, target_symbol, f"Error: {e}")

    def _create_fallback_result(self, state: PipelineState, symbol: str, reason: str) -> CriticResult:
        """创建兜底的批评结果 (默认为有效，但标记为弱)。"""
        return CriticResult(
            timestamp=state.current_time,
            target_symbol=symbol,
            is_valid=False, # [Safety] Default to REJECT on failure
            critique_summary=f"System Fallback (Validation Failed): {reason}",
            score=0.0, # Penalize score
            flaws_found=[],
            suggestions=[]
        )

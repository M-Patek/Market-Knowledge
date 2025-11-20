import json
import logging
from typing import List, Any, AsyncGenerator

from Phoenix_project.agents.l2.base import L2Agent
from Phoenix_project.core.schemas.evidence_schema import EvidenceItem
from Phoenix_project.core.schemas.adversary_result import AdversaryResult
from Phoenix_project.core.pipeline_state import PipelineState
from pydantic import ValidationError

logger = logging.getLogger(__name__)

class AdversaryAgent(L2Agent):
    """
    L2 智能体：对抗者 (Adversary)
    通过提出反面观点和压力测试来挑战主流共识（红队测试）。
    """

    async def run(self, state: PipelineState, dependencies: List[Any]) -> AsyncGenerator[AdversaryResult, None]:
        """
        [Refactored Phase 3.2] 适配 PipelineState，增加鲁棒性和兜底机制。
        """
        target_symbol = state.main_task_query.get("symbol", "UNKNOWN")
        logger.info(f"[{self.agent_id}] Running Adversary analysis for: {target_symbol}")

        # 1. 提取证据 (Input Tolerance)
        evidence_items = []
        if dependencies:
            for item in dependencies:
                if isinstance(item, EvidenceItem):
                    evidence_items.append(item)
                elif isinstance(item, dict) and "content" in item:
                    try:
                        evidence_items.append(EvidenceItem(**item))
                    except:
                        pass

        if not evidence_items:
            logger.warning(f"[{self.agent_id}] No evidence to challenge. Yielding fallback.")
            yield self._create_fallback_result(state, target_symbol, "No evidence provided.")
            return

        agent_prompt_name = "l2_adversary"

        try:
            # 2. 准备 Prompt
            evidence_json_list = json.dumps(
                [item.model_dump() for item in evidence_items], 
                default=str
            )
            
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
            
            # 强制时间同步
            if "timestamp" not in response_data:
                response_data["timestamp"] = state.current_time

            result = AdversaryResult.model_validate(response_data)
            object.__setattr__(result, 'timestamp', state.current_time)

            logger.info(f"[{self.agent_id}] Generated counter-arguments. Threat Level: {result.threat_level}")
            yield result

        except Exception as e:
            logger.error(f"[{self.agent_id}] Error: {e}", exc_info=True)
            yield self._create_fallback_result(state, target_symbol, f"Error: {e}")

    def _create_fallback_result(self, state: PipelineState, symbol: str, reason: str) -> AdversaryResult:
        """创建兜底的对抗结果 (LOW threat)。"""
        return AdversaryResult(
            timestamp=state.current_time,
            target_symbol=symbol,
            counter_argument=f"System Fallback: {reason}",
            threat_level="LOW",
            confidence=0.0,
            attack_vectors=[],
            challenged_evidence_ids=[]
        )

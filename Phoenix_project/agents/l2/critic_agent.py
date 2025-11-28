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
            
            # [Task 2.2] Explicit Mapping & Logic Layer
            # Handle potential list response or single object
            raw_items = response_data if isinstance(response_data, list) else [response_data]
            
            for raw in raw_items:
                # 1. Map Scores
                q_score = float(raw.get("quality_score", 0.0))
                c_score = float(raw.get("clarity_score", 0.0))
                b_score = float(raw.get("bias_score", 0.0))
                r_score = float(raw.get("relevance_score", 0.0))
                
                # 2. Derive Logic
                # Simple Average for validation (Assumption: bias_score is 'Low Bias' or handled positively)
                avg_score = (q_score + c_score + b_score + r_score) / 4.0
                is_valid = avg_score > 0.70
                
                # Derive Confidence Adjustment (Center at 0.5: >0.5 boosts, <0.5 penalizes)
                conf_adj = (avg_score - 0.5) * 0.4

                result = CriticResult(
                    agent_id=self.agent_id,
                    target_evidence_id=raw.get("original_evidence_id", "UNKNOWN"),
                    is_valid=is_valid,
                    critique=raw.get("critique", "No critique provided."),
                    confidence_adjustment=conf_adj,
                    quality_score=q_score,
                    clarity_score=c_score,
                    bias_score=b_score,
                    relevance_score=r_score,
                    suggestions=raw.get("suggestions_for_improvement", "")
                )
                # Inject timestamp for pipeline tracing
                object.__setattr__(result, 'timestamp', state.current_time)

                logger.info(f"[{self.agent_id}] Critique complete. Valid: {is_valid} (Avg: {avg_score:.2f})")
                yield result

        except Exception as e:
            logger.error(f"[{self.agent_id}] Error: {e}", exc_info=True)
            yield self._create_fallback_result(state, target_symbol, f"Error: {e}")

    def _create_fallback_result(self, state: PipelineState, symbol: str, reason: str) -> CriticResult:
        """创建兜底的批评结果 (默认为有效，但标记为弱)。"""
        return CriticResult(
            agent_id=self.agent_id,
            target_evidence_id="UNKNOWN",
            is_valid=False,
            critique=f"System Fallback: {reason}",
            confidence_adjustment=-0.5,
            suggestions="Check System Logs",
            flags=["SYSTEM_ERROR"]
        )

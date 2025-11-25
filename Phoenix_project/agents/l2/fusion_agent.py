import json
import logging
import re
from typing import List, Any, AsyncGenerator, Optional

from Phoenix_project.agents.l2.base import L2Agent
from Phoenix_project.core.schemas.evidence_schema import EvidenceItem
from Phoenix_project.core.schemas.fusion_result import FusionResult
from Phoenix_project.core.pipeline_state import PipelineState
from pydantic import ValidationError

# 获取日志记录器
logger = logging.getLogger(__name__)

class FusionAgent(L2Agent):
    """
    L2 智能体：融合 (Fusion)
    审查所有 L1 和 L2 的证据，融合冲突信息，并得出最终的 L2 决策。
    """

    async def run(self, state: PipelineState, dependencies: List[Any]) -> AsyncGenerator[FusionResult, None]:
        """
        异步运行智能体，以融合所有 L1/L2 证据。
        此智能体运行一次（不使用 gather），产出一个 FusionResult。
        [Refactored Phase 3.2] 适配 PipelineState，增加鲁棒性和兜底机制。
        
        参数:
            state (PipelineState): 当前管道状态 (提供时间与上下文)。
            dependencies (List[Any]): 上游 L1/L2 的输出列表 (EvidenceItem)。

        收益:
            AsyncGenerator[FusionResult, None]: 异步生成 *单一* 的 FusionResult 对象。
        """
        # 1. 确定目标 Symbol
        target_symbol = state.main_task_query.get("symbol", "UNKNOWN")
        task_desc = state.main_task_query.get("description", "Fusion Task")
        
        logger.info(f"[{self.agent_id}] Running FusionAgent for symbol: {target_symbol}")

        # 2. 宽容地提取证据 (Input Tolerance)
        evidence_items = []
        if dependencies:
            for item in dependencies:
                if isinstance(item, EvidenceItem):
                    evidence_items.append(item)
                elif isinstance(item, dict) and "content" in item:
                    # 尝试从字典恢复 (如果被序列化过)
                    try:
                        evidence_items.append(EvidenceItem(**item))
                    except Exception as e:
                        logger.warning(f"[{self.agent_id}] Failed to recover evidence item: {e}")

        if not evidence_items:
            logger.warning(f"[{self.agent_id}] No valid EvidenceItems found. Yielding fallback result.")
            yield self._create_fallback_result(state, target_symbol, "No evidence provided.")
            return

        logger.info(f"[{self.agent_id}] Found {len(evidence_items)} total evidence items (L1+L2) to fuse.")

        agent_prompt_name = "l2_fusion"

        try:
            # 3. 准备 Prompt 上下文
            # [Safety] Use _safe_prepare_context to prevent context explosion
            evidence_json_list = self._safe_prepare_context([item.model_dump() for item in evidence_items])
            
            context_map = {
                "symbols_list_str": target_symbol,
                "evidence_json_list": evidence_json_list
            }

            # 4. 异步调用 LLM
            logger.debug(f"[{self.agent_id}] Calling LLM with prompt: {agent_prompt_name} for fusion task.")
            
            response_str = await self.llm_client.run_llm_task(
                agent_prompt_name=agent_prompt_name,
                context_map=context_map
            )

            if not response_str:
                logger.warning(f"[{self.agent_id}] LLM returned empty response.")
                yield self._create_fallback_result(state, target_symbol, "LLM returned empty response.")
                return

            # 5. 解析和验证 FusionResult
            logger.debug(f"[{self.agent_id}] Received LLM fusion response (raw): {response_str[:200]}...")
            
            # [Robustness] Clean Markdown code blocks and extract JSON
            clean_str = response_str.strip()
            # Regex to find the JSON object between braces, ignoring surrounding text/markdown
            match = re.search(r"(\{[\s\S]*\})", clean_str)
            if match:
                clean_str = match.group(1)

            response_data = json.loads(clean_str)
            
            # [Robustness] Fuzzy Parsing & Normalization
            # 1. Map missing decision from sentiment
            if "decision" not in response_data:
                sentiment = response_data.get("overall_sentiment") or response_data.get("sentiment")
                if sentiment:
                    s_up = str(sentiment).upper()
                    if "BULL" in s_up or "POS" in s_up:
                        response_data["decision"] = "BUY"
                    elif "BEAR" in s_up or "NEG" in s_up:
                        response_data["decision"] = "SELL"
                    else:
                        response_data["decision"] = "HOLD"
                else:
                    response_data["decision"] = "HOLD"

            # [Fix Phase II] Synthesize numeric sentiment from decision if missing
            # 防止 L3 看到 "0.0" 的情感值即使决策是 "STRONG BUY"
            if "sentiment" not in response_data:
                d = response_data.get("decision", "HOLD").upper()
                if "STRONG_BUY" in d: response_data["sentiment"] = 1.0
                elif "BUY" in d: response_data["sentiment"] = 0.5
                elif "STRONG_SELL" in d: response_data["sentiment"] = -1.0
                elif "SELL" in d: response_data["sentiment"] = -0.5
                else: response_data["sentiment"] = 0.0

            # 2. Ensure confidence is float
            if "confidence" in response_data:
                try:
                    response_data["confidence"] = float(response_data["confidence"])
                except (ValueError, TypeError):
                    response_data["confidence"] = 0.0

            # 3. Defaults for list fields
            if "supporting_evidence_ids" not in response_data:
                response_data["supporting_evidence_ids"] = []
            
            # [Time Machine] 强制使用仿真时间
            if "timestamp" not in response_data:
                response_data["timestamp"] = state.current_time
            
            fusion_result = FusionResult.model_validate(response_data)
            # 再次确保时间一致性 (防止 model_validate 使用默认的 utcnow)
            object.__setattr__(fusion_result, 'timestamp', state.current_time)
            
            logger.info(f"[{self.agent_id}] Successfully generated FusionResult. Decision: {fusion_result.decision}")
            
            yield fusion_result

        except json.JSONDecodeError as e:
            logger.error(f"[{self.agent_id}] Failed to decode LLM JSON response for fusion task. Error: {e}")
            yield self._create_fallback_result(state, target_symbol, f"JSON Decode Error: {e}")
            
        except ValidationError as e:
            logger.error(f"[{self.agent_id}] Failed to validate FusionResult schema. Error: {e}")
            yield self._create_fallback_result(state, target_symbol, f"Schema Validation Error: {e}")
            
        except Exception as e:
            logger.error(f"[{self.agent_id}] An unexpected error occurred during fusion task. Error: {e}")
            yield self._create_fallback_result(state, target_symbol, f"Unexpected Error: {e}")

    def _create_fallback_result(self, state: PipelineState, symbol: str, reason: str) -> FusionResult:
        """
        [Resilience] 创建一个兜底的 FusionResult (NEUTRAL)，防止管道崩溃。
        """
        logger.warning(f"[{self.agent_id}] Creating FALLBACK FusionResult. Reason: {reason}")
        return FusionResult(
            timestamp=state.current_time,
            target_symbol=symbol,
            decision="NEUTRAL",
            confidence=0.0,
            reasoning=f"System Fallback: {reason}",
            uncertainty=1.0,
            supporting_evidence_ids=[],
            conflicting_evidence_ids=[]
        )

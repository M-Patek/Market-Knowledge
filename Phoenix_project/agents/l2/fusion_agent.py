import json
import logging
import re
from typing import List, Any, Optional

from Phoenix_project.agents.l2.base import L2Agent
from Phoenix_project.core.schemas.evidence_schema import EvidenceItem
# [Phase III Fix] Import SystemStatus
from Phoenix_project.core.schemas.fusion_result import FusionResult, SystemStatus
from Phoenix_project.core.pipeline_state import PipelineState
from pydantic import ValidationError

# 获取日志记录器
logger = logging.getLogger(__name__)

class FusionAgent(L2Agent):
    """
    L2 智能体：融合 (Fusion)
    审查所有 L1 和 L2 的证据，融合冲突信息，并得出最终的 L2 决策。
    [Code Opt Expert Fix] Task 08 & 09: DoS Prevention & Prompt Injection Sanitization
    """

    def _sanitize_symbol(self, symbol: str) -> str:
        """
        [Task 09] Sanitize symbol input to prevent Prompt Injection.
        Strictly allows alphanumeric, commas, dashes, and spaces.
        """
        if not symbol: return "UNKNOWN"
        # Remove any character that is NOT alphanumeric, space, comma, or dash
        return re.sub(r'[^a-zA-Z0-9\s,\-]', '', str(symbol)).strip()

    def _sanitize_general_input(self, text: str) -> str:
        """
        [Task 09] Sanitize general text inputs.
        Escapes XML-like tags to prevent system prompt overriding.
        """
        if not text: return ""
        return str(text).replace("<", "&lt;").replace(">", "&gt;")

    async def run(self, state: PipelineState, dependencies: List[Any]) -> FusionResult:
        """
        异步运行智能体，以融合所有 L1/L2 证据。
        [Task 1.3 Fix] Changed from AsyncGenerator to Direct Return.
        
        参数:
            state (PipelineState): 当前管道状态 (提供时间与上下文)。
            dependencies (List[Any]): 上游 L1/L2 的输出列表 (EvidenceItem)。

        收益:
            FusionResult: 返回 *单一* 的 FusionResult 对象。
        """
        # 1. 确定目标 Symbol (Task 09 Sanitization)
        raw_symbol = state.main_task_query.get("symbol", "UNKNOWN")
        target_symbol = self._sanitize_symbol(raw_symbol)
        
        raw_desc = state.main_task_query.get("description", "Fusion Task")
        task_desc = self._sanitize_general_input(raw_desc)
        
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
            logger.warning(f"[{self.agent_id}] No valid EvidenceItems found. Returning fallback result.")
            return self._create_fallback_result(state, target_symbol, "No evidence provided.")

        logger.info(f"[{self.agent_id}] Found {len(evidence_items)} total evidence items (L1+L2) to fuse.")

        agent_prompt_name = "l2_fusion"

        try:
            # 3. 准备 Prompt 上下文
            # [Safety] Use _safe_prepare_context to prevent context explosion
            evidence_json_list = self._safe_prepare_context([item.model_dump() for item in evidence_items])
            
            context_map = {
                "symbols_list_str": target_symbol,
                "evidence_json_list": evidence_json_list,
                "task_desc": task_desc # Pass sanitized description
            }

            # 4. 异步调用 LLM
            logger.debug(f"[{self.agent_id}] Calling LLM with prompt: {agent_prompt_name} for fusion task.")
            
            response_str = await self.llm_client.run_llm_task(
                agent_prompt_name=agent_prompt_name,
                context_map=context_map
            )

            if not response_str:
                logger.warning(f"[{self.agent_id}] LLM returned empty response.")
                return self._create_fallback_result(state, target_symbol, "LLM returned empty response.")

            # 5. 解析和验证 FusionResult
            logger.debug(f"[{self.agent_id}] Received LLM fusion response (raw): {response_str[:200]}...")
            
            # [Robustness] Clean Markdown code blocks and extract JSON
            clean_str = response_str.strip()
            
            # [Task 08] Optimized JSON Extraction (DoS Prevention)
            # Efficiently locate the outermost JSON boundaries
            start_idx = clean_str.find('{')
            end_idx = clean_str.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                clean_str = clean_str[start_idx : end_idx + 1]
            else:
                # Fallback to simple strip if braces not found or malformed
                pass

            response_data = json.loads(clean_str)
            
            # [Robustness] Fuzzy Parsing & Normalization
            # 1. Map missing decision from sentiment
            if "decision" not in response_data:
                sentiment = response_data.get("overall_sentiment") or response_data.get("sentiment")
                if sentiment:
                    s_up = str(sentiment).upper().replace(" ", "_")
                    if "BULL" in s_up or "POS" in s_up:
                        response_data["decision"] = "BUY"
                    elif "BEAR" in s_up or "NEG" in s_up:
                        response_data["decision"] = "SELL"
                    else:
                        response_data["decision"] = "HOLD"
                else:
                    response_data["decision"] = "HOLD"

            # [Fix Phase II] Synthesize numeric sentiment from decision if missing
            if "sentiment" not in response_data:
                # [Task 6.2 Fix] Externalized sentiment mapping
                decision_key = response_data.get("decision", "HOLD").upper().replace(" ", "_")
                
                # Load mapping from config or use safe defaults
                default_mapping = {"STRONG_BUY": 1.0, "BUY": 0.5, "HOLD": 0.0, "NEUTRAL": 0.0, "SELL": -0.5, "STRONG_SELL": -1.0}
                mapping = self.config.get("sentiment_mapping", default_mapping)
                
                response_data["sentiment"] = mapping.get(decision_key, 0.0)

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

            # [Phase III Fix] The Zero Trap Logic
            # Dynamically set system status based on confidence
            conf = response_data.get("confidence", 0.0)
            if conf < 0.1:
                response_data["system_status"] = SystemStatus.HALT
            elif conf < 0.3:
                response_data["system_status"] = SystemStatus.DEGRADED
            else:
                response_data["system_status"] = SystemStatus.OK
            
            fusion_result = FusionResult.model_validate(response_data)
            
            logger.info(f"[{self.agent_id}] Successfully generated FusionResult. Decision: {fusion_result.decision}")
            
            return fusion_result

        except json.JSONDecodeError as e:
            logger.error(f"[{self.agent_id}] Failed to decode LLM JSON response for fusion task. Error: {e}")
            return self._create_fallback_result(state, target_symbol, f"JSON Decode Error: {e}")
            
        except ValidationError as e:
            logger.error(f"[{self.agent_id}] Failed to validate FusionResult schema. Error: {e}")
            return self._create_fallback_result(state, target_symbol, f"Schema Validation Error: {e}")
            
        except Exception as e:
            logger.error(f"[{self.agent_id}] An unexpected error occurred during fusion task. Error: {e}")
            return self._create_fallback_result(state, target_symbol, f"Unexpected Error: {e}")

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
            # [Phase III Fix] Explicitly signal HALT on fallback
            system_status=SystemStatus.HALT,
            supporting_evidence_ids=[],
            conflicting_evidence_ids=[]
        )

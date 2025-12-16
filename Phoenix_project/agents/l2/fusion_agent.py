import json
import logging
import re
from typing import List, Any, Optional, Dict

from Phoenix_project.agents.l2.base import L2Agent
from Phoenix_project.core.schemas.evidence_schema import EvidenceItem
# [Phase III Fix] Import SystemStatus
from Phoenix_project.core.schemas.fusion_result import FusionResult, SystemStatus
from Phoenix_project.core.pipeline_state import PipelineState
# [Task FIX-MED-002] Import utility
from Phoenix_project.utils import extract_json_from_text
from pydantic import ValidationError

# 获取日志记录器
logger = logging.getLogger(__name__)

class FusionAgent(L2Agent):
    """
    L2 智能体：融合 (Fusion)
    审查所有 L1 和 L2 的证据，融合冲突信息，并得出最终的 L2 决策。
    [Code Opt Expert Fix] Task 08 & 09: DoS Prevention & Prompt Injection Sanitization
    [P1-RISK-02] Anti-Hallucination & Rule Validation
    """

    def _sanitize_symbol(self, symbol: str) -> str:
        """
        [Task 09] Sanitize symbol input to prevent Prompt Injection.
        Strictly allows alphanumeric, commas, dashes, and spaces.
        """
        if not symbol: return "UNKNOWN"
        # Remove any character that is NOT alphanumeric, space, comma, or dash
        # [Task 3.1 Fix] Allow forward slash and dot for Crypto/Stock tickers (BTC/USD, BRK.B)
        return re.sub(r'[^a-zA-Z0-9\s,\-/\.]', '', str(symbol)).strip()

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
        
        # [Task P1-RISK-02] Retry loop for JSON errors
        max_retries = 2
        
        for attempt in range(max_retries + 1):
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
                logger.debug(f"[{self.agent_id}] Calling LLM (Attempt {attempt+1})...")
                
                response_str = await self.llm_client.run_llm_task(
                    agent_prompt_name=agent_prompt_name,
                    context_map=context_map
                )

                if not response_str:
                    logger.warning(f"[{self.agent_id}] LLM returned empty response.")
                    continue

                # 5. 解析和验证 FusionResult
                logger.debug(f"[{self.agent_id}] Received LLM fusion response (raw): {response_str[:200]}...")
                
                # [Task FIX-MED-002] Robustness: Use utility function
                clean_str = extract_json_from_text(response_str)

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
                    decision_key = response_data.get("decision", "HOLD").upper().replace(" ", "_")
                    
                    # [Task 5.2 Fix] Explicit Logic for Transparency
                    fallback_mapping = {"STRONG_BUY": 1.0, "BUY": 0.5, "HOLD": 0.0, "NEUTRAL": 0.0, "SELL": -0.5, "STRONG_SELL": -1.0}
                    mapping = self.config.get("sentiment_mapping")
                    
                    if not mapping:
                         # logger.warning(f"[{self.agent_id}] Sentiment mapping missing in config, using hardcoded fallback.")
                         mapping = fallback_mapping
                    
                    response_data["sentiment"] = mapping.get(decision_key, 0.0)

                # [Task P1-RISK-02] Rule-based Validation (Anti-Hallucination)
                validated_data = self._validate_with_rules(response_data, evidence_items)

                # 2. Ensure confidence is float
                if "confidence" in validated_data:
                    try:
                        validated_data["confidence"] = float(validated_data["confidence"])
                    except (ValueError, TypeError):
                        validated_data["confidence"] = 0.0

                # 3. Defaults for list fields
                if "supporting_evidence_ids" not in validated_data:
                    validated_data["supporting_evidence_ids"] = []
                
                # [Time Machine] 强制使用仿真时间
                if "timestamp" not in validated_data:
                    validated_data["timestamp"] = state.current_time

                # [Phase III Fix] The Zero Trap Logic
                # Dynamically set system status based on confidence
                conf = validated_data.get("confidence", 0.0)
                if conf < 0.1:
                    validated_data["system_status"] = SystemStatus.HALT
                elif conf < 0.3:
                    validated_data["system_status"] = SystemStatus.DEGRADED
                else:
                    validated_data["system_status"] = SystemStatus.OK
                
                fusion_result = FusionResult.model_validate(validated_data)
                
                logger.info(f"[{self.agent_id}] Successfully generated FusionResult. Decision: {fusion_result.decision}")
                
                return fusion_result

            except json.JSONDecodeError as e:
                logger.warning(f"[{self.agent_id}] JSON Decode Error (Attempt {attempt+1}): {e}")
                if attempt == max_retries:
                    return self._create_fallback_result(state, target_symbol, f"JSON Decode Error: {e}")
            
            except ValidationError as e:
                logger.error(f"[{self.agent_id}] Failed to validate FusionResult schema. Error: {e}")
                return self._create_fallback_result(state, target_symbol, f"Schema Validation Error: {e}")
            
            except Exception as e:
                logger.error(f"[{self.agent_id}] An unexpected error occurred during fusion task. Error: {e}")
                return self._create_fallback_result(state, target_symbol, f"Unexpected Error: {e}")
        
        return self._create_fallback_result(state, target_symbol, "Max retries exceeded.")

    def _validate_with_rules(self, llm_data: Dict[str, Any], evidence_items: List[EvidenceItem]) -> Dict[str, Any]:
        """
        [Task P1-RISK-02] Cross-verify LLM conclusion against statistical average of inputs.
        Prevents "Hallucinated Reversals" (e.g., Inputs All Bearish -> LLM Bullish).
        """
        try:
            # 1. Calculate Weighted Average of Input Sentiment
            total_weight = 0.0
            weighted_sentiment_sum = 0.0
            
            # [Fix] Comprehensive mapping
            s_map = {
                "STRONG_BUY": 1.0, "STRONG BUY": 1.0,
                "BUY": 0.5, "BULLISH": 0.5, "BULL": 0.5, "POSITIVE": 0.5,
                "HOLD": 0.0, "NEUTRAL": 0.0,
                "SELL": -0.5, "BEARISH": -0.5, "BEAR": -0.5, "NEGATIVE": -0.5,
                "STRONG_SELL": -1.0, "STRONG SELL": -1.0
            }

            for item in evidence_items:
                s_val = item.sentiment
                if isinstance(s_val, str):
                    key = str(s_val).upper().replace("_", " ").strip()
                    if key in s_map:
                        s_val = s_map[key]
                    elif "BULL" in key or "POS" in key:
                         s_val = 0.5
                    elif "BEAR" in key or "NEG" in key:
                         s_val = -0.5
                    else:
                         s_val = 0.0
                
                try:
                    s_val = float(s_val)
                except (ValueError, TypeError):
                    s_val = 0.0

                conf = item.confidence if item.confidence is not None else 0.5
                weighted_sentiment_sum += s_val * float(conf)
                total_weight += float(conf)
            
            if total_weight == 0:
                return llm_data 
            
            avg_input_sentiment = weighted_sentiment_sum / total_weight
            llm_sentiment = float(llm_data.get("sentiment", 0.0))
            
            # 2. Check Deviation
            deviation = abs(llm_sentiment - avg_input_sentiment)
            
            # [Fix] Threshold strictly set to 0.5
            if deviation > 0.5: 
                logger.warning(
                    f"[{self.agent_id}] HALLUCINATION WARNING: LLM Sentiment ({llm_sentiment}) deviates from Input Avg ({avg_input_sentiment:.2f}) by {deviation:.2f}."
                )
                
                # Penalty: Cap confidence and force Neutral if direction flipped
                llm_data["confidence"] = min(float(llm_data.get("confidence", 0.5)), 0.2)
                llm_data["reasoning"] = f"[AUTO-CORRECT] Original LLM reasoning flagged. Inputs Avg: {avg_input_sentiment:.2f}. " + str(llm_data.get("reasoning", ""))
                
                # If completely opposite signs (allowing small noise)
                if (avg_input_sentiment * llm_sentiment) < -0.05:
                     logger.warning(f"[{self.agent_id}] Overriding LLM Decision to NEUTRAL due to polarity mismatch.")
                     llm_data["decision"] = "NEUTRAL"
                     llm_data["sentiment"] = 0.0
                     
            return llm_data
            
        except Exception as e:
            logger.error(f"[{self.agent_id}] Rule validation failed: {e}")
            return llm_data

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

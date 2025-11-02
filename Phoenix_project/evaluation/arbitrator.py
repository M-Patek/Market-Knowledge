"""
仲裁者 (Arbitrator)
负责在多个 AI 智能体之间存在分歧时，做出最终决策。
"""
from typing import List, Dict, Any
from datetime import datetime
import uuid
# 恢复原有逻辑所需的新增导入
import json
from pydantic import ValidationError


# FIX (E3): 导入 AgentDecision 和 FusionResult (这个保留)
from core.schemas.fusion_result import AgentDecision, FusionResult
from ai.prompt_manager import PromptManager
from api.gateway import IAPIGateway
from monitor.logging import get_logger

logger = get_logger(__name__)

class Arbitrator:
    """
    使用一个专门的 "Arbitrator" LLM 提示，
    来审查所有分析师的观点，并生成一个最终的 FusionResult。
    """

    def __init__(self, api_gateway: IAPIGateway, prompt_manager: PromptManager):
        self.api_gateway = api_gateway
        self.prompt_manager = prompt_manager
        self.log_prefix = "Arbitrator:"

    def arbitrate(self, decisions: List[AgentDecision], context: str) -> FusionResult:
        """
        执行仲裁过程。
        """
        logger.info(f"{self.log_prefix} Arbitrating {len(decisions)} decisions...")
        
        # 1. 序列化决策
        serialized_decisions = self._serialize_decisions_for_prompt(decisions)
        
        # 2. 渲染仲裁者提示
        prompt = self.prompt_manager.render_prompt(
            "arbitrator", # 来自 prompts/arbitrator.json
            context=context,
            agent_decisions=serialized_decisions
        )
        
        if not prompt:
            logger.error(f"{self.log_prefix} Failed to render 'arbitrator' prompt.")
            # 返回一个紧急停止的结果
            return self._create_emergency_fusion_result(decisions, "Prompt rendering failed")

        # 3. 调用 LLM 并期望结构化输出
        try:
            # --- 恢复原有逻辑 ---
            # 1. 调用 LLM (假设返回文本)
            response_text = self.api_gateway.generate(
                prompt=prompt,
                model_id="gemini-1.5-pro" # 仲裁需要强大的模型
            )

            if not response_text:
                logger.error(f"{self.log_prefix} Arbitration LLM call returned empty response.")
                return self._create_emergency_fusion_result(decisions, "LLM returned empty response")

            # 2. 将响应文本 (JSON) 解析为字典
            try:
                # 尝试找到json代码块
                if "```json" in response_text:
                    json_text = response_text.split("```json")[1].split("```")[0].strip()
                else:
                    json_text = response_text
                
                response_data = json.loads(json_text)
            except json.JSONDecodeError:
                logger.error(f"{self.log_prefix} Arbitration LLM returned non-JSON response: {response_text}")
                return self._create_emergency_fusion_result(decisions, "LLM returned non-JSON response")
            
            # 3. 验证字典并转换为 Pydantic 模型
            try:
                fusion_result = FusionResult(**response_data)
            except ValidationError as e:
                logger.error(f"{self.log_prefix} Arbitration LLM returned invalid JSON structure: {e}\nResponse: {response_data}")
                return self._create_emergency_fusion_result(decisions, f"LLM returned invalid JSON structure: {e}")
            
            # --- 结束恢复原有逻辑 ---
            
            # 4. 填充 LLM 可能遗漏的字段
            fusion_result.agent_decisions = decisions # 确保原始决策被包括在内
            fusion_result.timestamp = datetime.utcnow()
            fusion_result.id = f"fusion_{uuid.uuid4()}"
            
            logger.info(f"{self.log_prefix} Arbitration complete. Final decision: {fusion_result.final_decision}")
            return fusion_result

        except Exception as e:
            logger.error(f"{self.log_prefix} Arbitration LLM call failed: {e}", exc_info=True)
            return self._create_emergency_fusion_result(decisions, str(e))

    def _serialize_decisions_for_prompt(self, decisions: List[AgentDecision]) -> str:
        """
        将 AgentDecision 列表转换为 LLM 提示符可读的格式。
        (与 MetacognitiveAgent 中的方法重复，可以提取到工具类中)
        """
        prompt_text = ""
        for i, decision in enumerate(decisions):
            prompt_text += f"\n--- Decision {i+1}: Agent '{decision.agent_name}' ---\n"
            prompt_text += f"Decision: {decision.decision} (Confidence: {decision.confidence*100:.0f}%)\n"
            prompt_text += "Reasoning:\n"
            prompt_text += decision.reasoning
            prompt_text += "\n----------------------------------\n"
        return prompt_text

    def _create_emergency_fusion_result(self, decisions: List[AgentDecision], error_msg: str) -> FusionResult:
        """
        在仲裁失败时，返回一个“中立/暂停”的结果。
        """
        return FusionResult(
            id=f"fusion_error_{uuid.uuid4()}",
            timestamp=datetime.utcnow(),
            agent_decisions=decisions,
            final_decision="NEUTRAL",
            final_confidence=0.5,
            summary=f"Arbitration failed: {error_msg}. Defaulting to NEUTRAL.",
            conflicts_identified=["Arbitration Failure"],
            conflict_resolution=None,
            uncertainty_score=1.0, # 最高不确定性
            uncertainty_dimensions={"arbitration_failure": 1.0}
        )


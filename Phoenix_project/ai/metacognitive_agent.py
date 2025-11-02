"""
元认知智能体 (Metacognitive Agent)
负责审查其他智能体的推理过程 (Chain of Thought, CoT)。
"""
from typing import List, Dict, Any, Optional

# FIX (E3): 导入 AgentDecision
from core.schemas.fusion_result import AgentDecision
from ai.prompt_manager import PromptManager
from api.gateway import IAPIGateway
from monitor.logging import get_logger

logger = get_logger(__name__)

class MetacognitiveAgent:
    """
    一个特殊的智能体，不进行市场分析，而是分析 *其他智能体* 的分析。
    """

    def __init__(self, api_gateway: IAPIGateway, prompt_manager: PromptManager):
        self.api_gateway = api_gateway
        self.prompt_manager = prompt_manager
        self.log_prefix = "MetacognitiveAgent:"

    def analyze_decisions(self, decisions: List[AgentDecision], context: str) -> Dict[str, Any]:
        """
        审查所有智能体的决策，识别分歧、缺失的上下文和潜在的幻觉。
        
        :param decisions: 来自 EnsembleClient 的 AgentDecision 列表。
        :param context: 提供给分析师的原始上下文。
        :return: 一个包含分析结果的字典。
        """
        
        logger.info(f"{self.log_prefix} Analyzing {len(decisions)} agent decisions...")
        
        # 1. 序列化决策以供 LLM 使用
        serialized_decisions = self._serialize_decisions_for_prompt(decisions)
        
        # 2. 获取元认知提示 (e.g., "critic" or "arbitrator" prompt)
        # 这里我们假设有一个 "critic" 提示
        prompt = self.prompt_manager.render_prompt(
            "critic", # 假设 "critic" 是元认知提示
            context=context,
            agent_decisions=serialized_decisions
        )
        
        if not prompt:
            logger.error(f"{self.log_prefix} Failed to render 'critic' prompt.")
            return {"error": "Failed to render prompt"}

        # 3. 调用 LLM
        try:
            # 元认知分析可能需要一个强大的模型
            analysis_text = self.api_gateway.generate(prompt, model_id="gemini-1.5-pro")
            
            # 4. (理想情况下) 将 analysis_text 解析为结构化输出
            # 例如：{ "conflicts": [...], "uncertainty_score": 0.7, "summary": "..." }
            # 为简单起见，我们只返回原始文本
            
            logger.info(f"{self.log_prefix} Analysis complete.")
            return {"analysis_summary": analysis_text}

        except Exception as e:
            logger.error(f"{self.log_prefix} Analysis failed: {e}", exc_info=True)
            return {"error": str(e)}

    def _serialize_decisions_for_prompt(self, decisions: List[AgentDecision]) -> str:
        """
        将 AgentDecision 列表转换为 LLM 提示符可读的格式。
        """
        prompt_text = ""
        for i, decision in enumerate(decisions):
            prompt_text += f"\n--- Decision {i+1}: Agent '{decision.agent_name}' ---\n"
            prompt_text += f"Decision: {decision.decision} (Confidence: {decision.confidence*100:.0f}%)\n"
            prompt_text += "Reasoning:\n"
            prompt_text += decision.reasoning
            prompt_text += "\n----------------------------------\n"
        return prompt_text

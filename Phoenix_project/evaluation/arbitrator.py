from typing import List, Dict, Any

from ..ai.ensemble_client import EnsembleClient
from ..ai.prompt_manager import PromptManager
from ..ai.prompt_renderer import PromptRenderer
from ..core.schemas.supervision_result import SupervisionResult
from ..core.schemas.fusion_result import FusionResult
from ..core.schemas.data_schema import Sentiment
from ..monitor.logging import get_logger

# Refactor: 导入共享的序列化函数，以消除代码重复
from ..agents.l2.metacognitive_agent import MetacognitiveAgent

logger = get_logger(__name__)

class Arbitrator:
    """
    Arbitrator (仲裁者) 负责在 L2 智能体的决策（例如来自 Critic 和 Adversary）
    与 L1 智能体（Fusion）的原始输出之间进行最终决策。
    它充当一个最终的、基于规则或模型的检查，以确保系统的一致性和安全性。
    """

    def __init__(self, 
                 llm_client: EnsembleClient, 
                 prompt_manager: PromptManager, 
                 prompt_renderer: PromptRenderer):
        self.llm_client = llm_client
        self.prompt_manager = prompt_manager
        self.prompt_renderer = prompt_renderer
        self.prompt = self.prompt_manager.get_prompt("arbitrator")
        self.model_name = "gemini-1.5-pro-latest" # 假设使用一个强大的模型进行仲裁
        logger.info("Arbitrator initialized.")

    async def arbitrate(
        self, 
        original_fusion: FusionResult, 
        decisions: List[SupervisionResult]
    ) -> SupervisionResult:
        """
        根据 L2 智能体的监督决策和原始的 L1 融合结果，做出最终裁决。

        Args:
            original_fusion: L1 融合智能体的原始输出。
            decisions: L2 智能体（如 Critic, Adversary）的 SupervisionResult 列表。

        Returns:
            一个最终的 SupervisionResult，代表仲裁后的决策。
        """
        logger.info(f"Arbitrating {len(decisions)} decisions for symbol {original_fusion.symbol}...")
        
        # --- Refactor: 使用从 MetacognitiveAgent 导入的共享静态方法 ---
        # 移除了本地重复的 _serialize_decisions_for_prompt 方法
        serialized_decisions = MetacognitiveAgent.serialize_decisions_for_prompt(decisions)
        # --- End Refactor ---

        prompt_context = {
            "original_sentiment": original_fusion.sentiment.value,
            "original_confidence": original_fusion.confidence,
            "supervision_decisions": serialized_decisions
        }
        
        try:
            prompt = self.prompt_renderer.render(self.prompt, **prompt_context)
            
            response = await self.llm_client.run_chain(
                prompt,
                model_name=self.model_name,
                # Arbitrator 通常不解析工具，除非它需要核查事实
            )
            
            # 解析 LLM 的仲裁响应
            # 假设 LLM 返回一个类似于 SupervisionResult 的结构 (e.g., JSON)
            # 在一个真实的实现中，这里会有健壮的 JSON 解析和验证
            
            # --- 模拟解析 ---
            # 这是一个简化的解析逻辑。
            # 假设LLM的文本输出是："decision: OVERRIDE, final_sentiment: NEGATIVE, confidence: 0.85, reason: ..."
            # 我们将使用一个简化的逻辑：如果 L2 决策中存在强烈的反对意见，则进行否决。
            
            return self._rule_based_arbitration(original_fusion, decisions)

        except Exception as e:
            logger.error(f"Error during LLM arbitration: {e}. Falling back to rule-based arbitration.", exc_info=True)
            # 回退到基于规则的仲裁
            return self._rule_based_arbitration(original_fusion, decisions)

    def _rule_based_arbitration(
        self, 
        original_fusion: FusionResult, 
        decisions: List[SupervisionResult]
    ) -> SupervisionResult:
        """
        一个基于规则的仲裁回退逻辑。
        """
        logger.info("Applying rule-based arbitration fallback...")
        
        # 寻找强烈的否决票 (e.g., Critic or Adversary 提出了高置信度的反对意见)
        for decision in decisions:
            if (decision.decision.action == "OVERRIDE" or 
                decision.decision.action == "FLAG_FOR_REVIEW") and \
               decision.decision.confidence > 0.75:
                
                # 如果有强烈的否决票，采取否决票的意见
                final_decision = decision.decision
                final_decision.reason = f"[Arbitrator_Rule] Overridden by {decision.source_agent}: {final_decision.reason}"
                logger.warning(f"Rule-based arbitration: Overriding original fusion based on {decision.source_agent}.")
                return SupervisionResult(
                    source_agent="Arbitrator",
                    decision=final_decision,
                    timestamp=asyncio.get_event_loop().time()
                )
                
        # 如果没有强烈的否决票，则维持 L1 的融合结果
        logger.info("Rule-based arbitration: No strong objections found. Maintaining L1 fusion.")
        return SupervisionResult(
            source_agent="Arbitrator",
            decision={
                "action": "MAINTAIN",
                "final_sentiment": original_fusion.sentiment,
                "confidence": original_fusion.confidence,
                "reason": "L1 fusion maintained. No strong L2 objections."
            },
            timestamp=asyncio.get_event_loop().time()
        )

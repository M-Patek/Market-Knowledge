import asyncio  # 导入 asyncio
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
        
        # --- 优化：更新提示 ---
        # 加载 'l2_fusion' 提示，与核心 L2 FusionAgent 保持一致
        # 旧值: "arbitrator"
        self.prompt = self.prompt_manager.get_prompt("l2_fusion")
        # --- 结束优化 ---
        
        self.model_name = "gemini-1.5-pro-latest" # 假设使用一个强大的模型进行仲裁
        
        if not self.prompt:
             logger.error("Failed to load 'l2_fusion' prompt for Arbitrator. Check prompts directory.", exc_info=True)
             raise FileNotFoundError("Arbitrator prompt 'l2_fusion' not found.")
        
        logger.info("Arbitrator initialized with 'l2_fusion' prompt.")

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
            "supervision_decisions": serialized_decisions,
            # 确保传递 'l2_fusion' 提示可能需要的其他上下文
            "symbol": original_fusion.symbol,
            "evidence_items": "...", # 注意：可能需要传递 L1 证据的序列化版本
            "criticisms": serialized_decisions # 融合提示可能期望以 'criticisms' 命名
        }
        
        try:
            # 确保 prompt_context 包含了 'l2_fusion' 提示所需的所有变量
            # 注意：'l2_fusion' 提示可能比旧的 'arbitrator' 提示需要更多的上下文
            prompt_str = self.prompt_renderer.render(self.prompt, **prompt_context)
            
            # [Task 2.3] Temporarily disable LLM arbitration until structured parsing is implemented
            # response = await self.llm_client.run_chain(
            #     prompt_str,
            #     model_name=self.model_name,
            #     # Arbitrator 通常不解析工具，除非它需要核查事实
            # )
            
            # TODO: Implement structured output parsing for LLM arbitration.
            # Currently falling back to rule-based to save tokens and avoid parsing errors.
            
            logger.info("LLM arbitration skipped (not implemented). Using rule-based fallback.")
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
        
        current_time = asyncio.get_event_loop().time() # 获取当前时间

        # 寻找强烈的否决票 (e.g., Critic or Adversary 提出了高置信度的反对意见)
        for decision in decisions:
            # [Task 1.2] Fix: Access Pydantic model attributes directly
            if (decision.decision.action == "OVERRIDE" or 
                decision.decision.action == "FLAG_FOR_REVIEW") and \
               decision.decision.confidence > 0.75:
                
                # 如果有强烈的否决票，采取否决票的意见
                final_decision = decision.decision.model_dump() # [Task 1.2] Use model_dump()
                final_decision["reason"] = f"[Arbitrator_Rule] Overridden by {decision.source_agent}: {final_decision.get('reason', 'N/A')}"
                logger.warning(f"Rule-based arbitration: Overriding original fusion based on {decision.source_agent}.")
                return SupervisionResult(
                    source_agent="Arbitrator",
                    decision=final_decision,
                    timestamp=current_time
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
            timestamp=current_time
        )

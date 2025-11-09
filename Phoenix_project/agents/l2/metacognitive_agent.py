import asyncio
from typing import List, Dict, Any

from .base import L2Agent
from ...core.schemas.data_schema import Sentiment
from ...core.schemas.supervision_result import SupervisionResult, Decision
from ...monitor.logging import get_logger

logger = get_logger(__name__)

class MetacognitiveAgent(L2Agent):
    """
    元认知智能体 (Metacognitive Agent) 
    
    L2
    
    它负责监督 L1 智能体的输出和 L2 智能体（如 Critic 和 Adversary）的决策。
    它的目标是评估整个系统的推理过程，识别潜在的认知偏差、
    循环逻辑或 L2 监督本身的问题。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prompt = self.prompt_manager.get_prompt("l2_metacognitive") # 假设有一个特定的提示
        logger.info(f"MetacognitiveAgent initialized with model: {self.model_name}")

    # --- Refactor: 标记为 @staticmethod 并设为 public ---
    # 将此方法标记为静态方法，并使其公开 (public)，
    # 这样其他模块 (如 Arbitrator) 就可以导入并重用它，
    # 从而消除了代码重复。
    @staticmethod
    def serialize_decisions_for_prompt(decisions: List[SupervisionResult]) -> str:
        """
        将一系列 SupervisionResult 决策序列化为用于 LLM 提示的字符串。
        这是一个共享的工具函数。
        """
        if not decisions:
            return "No supervision decisions were provided."
        
        serialized = []
        for i, res in enumerate(decisions):
            dec_str = (
                f"Decision {i+1} (from {res.source_agent}):\n"
                f"- Action: {res.decision.action}\n"
                f"- Sentiment: {res.decision.final_sentiment.value if res.decision.final_sentiment else 'N/A'}\n"
                f"- Confidence: {res.decision.confidence:.2f}\n"
                f"- Reason: {res.decision.reason}\n"
            )
            serialized.append(dec_str)
            
        return "\n".join(serialized)
    # --- End Refactor ---

    async def run(self, 
                l1_analyses: List[Dict[str, Any]], 
                l2_decisions: List[SupervisionResult]) -> SupervisionResult:
        """
        执行元认知分析。

        Args:
            l1_analyses: L1 智能体的原始分析列表。
            l2_decisions: L2 智能体（Critic, Adversary等）的决策列表。

        Returns:
            一个 SupervisionResult，评估整个 L1 和 L2 过程。
        """
        logger.info(f"MetacognitiveAgent running analysis on {len(l1_analyses)} L1 analyses and {len(l2_decisions)} L2 decisions.")
        
        try:
            # 序列化 L1 分析 (简化)
            l1_summary = [f"Agent {d.get('agent_id', 'Unknown')}: {d.get('insight', 'No insight')[:100]}..." 
                          for d in l1_analyses]
            l1_summary_str = "\n".join(l1_summary)

            # --- Refactor: 使用静态方法 ---
            l2_decisions_str = self.serialize_decisions_for_prompt(l2_decisions)
            # --- End Refactor ---

            prompt_context = {
                "l1_analyses_summary": l1_summary_str,
                "l2_supervision_decisions": l2_decisions_str
            }
            
            prompt = self.prompt_renderer.render(self.prompt, **prompt_context)
            
            response_json = await self.llm_client.run_chain_structured(
                prompt,
                model_name=self.model_name
            )
            
            # 假设 response_json 遵循 Decision Pydantic 模型的结构
            decision = Decision(**response_json)
            
            return SupervisionResult(
                source_agent=self.agent_id,
                decision=decision,
                timestamp=asyncio.get_event_loop().time()
            )

        except Exception as e:
            logger.error(f"MetacognitiveAgent failed: {e}", exc_info=True)
            return SupervisionResult(
                source_agent=self.agent_id,
                decision=Decision(
                    action="FLAG_FOR_REVIEW",
                    final_sentiment=None,
                    confidence=0.9, # 对失败的置信度很高
                    reason=f"Metacognitive analysis failed due to exception: {e}"
                ),
                timestamp=asyncio.get_event_loop().time()
            )

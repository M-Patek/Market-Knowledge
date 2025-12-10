import math
from typing import List, Union
import asyncio

from ..core.schemas.supervision_result import SupervisionResult, Decision
from ..core.schemas.fusion_result import FusionResult
from ..core.schemas.data_schema import Sentiment
from ..monitor.logging import get_logger

logger = get_logger(__name__)

class Voter:
    """
    Voter (投票者) 负责聚合来自多个 L2 智能体（如 Critic, Adversary）
    的监督决策 (SupervisionResult)。
    
    它使用一种投票或融合机制来产生一个单一的、统一的 L2 监督决策。
    
    ---
    
    重构说明：
    此实现已从一个简化的占位符更新为一个使用
    "对数几率 (Log-Odds) 融合" 的更健壮的贝叶斯方法。
    
    这种方法将每个决策的置信度（概率）转换为对数几率，
    将它们相加（这是在对数空间中乘以概率），
    然后将最终的对数几率转换回概率。
    
    这是一种在贝叶斯推理中组合独立证据的标准方法。
    """

    def __init__(self, neutral_threshold: float = 0.1):
        """
        初始化投票者。
        
        Args:
            neutral_threshold: 用于定义中性区域的阈值 (围绕 0.5 概率)。
                例如，0.1 表示 0.45 到 0.55 之间的概率被视为中性。
        """
        self.neutral_threshold = neutral_threshold
        logger.info(f"Voter initialized with Bayesian Log-Odds Fusion (neutral_threshold={neutral_threshold}).")
        
    def _to_log_odds(self, p: float) -> float:
        """将概率 [0, 1] 转换为对数几率 [-inf, inf]。"""
        # 裁剪以避免 log(0) 或除以 0
        epsilon = 1e-9
        p = max(epsilon, min(1.0 - epsilon, p))
        return math.log(p / (1.0 - p))

    def _from_log_odds(self, log_odds: float) -> float:
        """将对数几率 [-inf, inf] 转换回概率 [0, 1]。"""
        try:
            return 1.0 / (1.0 + math.exp(-log_odds))
        except OverflowError:
            # 如果 log_odds 极大或极小
            return 1.0 if log_odds > 0 else 0.0

    async def vote(self, decisions: List[Union[SupervisionResult, FusionResult]]) -> SupervisionResult:
        """
        使用贝叶斯融合逻辑聚合 L2 决策。
        [Fix] Supports FusionResult inputs to prevent AttributeErrors.

        Args:
            decisions: L2 智能体的 SupervisionResult 或 FusionResult 列表。

        Returns:
            一个代表 L2 统一意见的 SupervisionResult。
        """
        if not decisions:
            logger.warning("Voter received no decisions to vote on. Returning NEUTRAL.")
            return SupervisionResult(
                source_agent="Voter",
                decision=Decision(
                    action="MAINTAIN",
                    final_sentiment=Sentiment.NEUTRAL,
                    confidence=0.0,
                    reason="No L2 decisions provided for voting."
                ),
                timestamp=asyncio.get_event_loop().time()
            )

        logger.info(f"Voter aggregating {len(decisions)} L2 decisions...")

        # 假设先验概率为 50/50 (P(Pos)=0.5)，其对数几率为 log(0.5/0.5) = 0
        total_log_odds = 0.0
        total_confidence_weight = 0.0
        
        override_reasons = []
        final_action_flags = set()

        for res in decisions:
            # --- Adapter Logic for FusionResult ---
            if isinstance(res, FusionResult):
                confidence = res.confidence
                # Convert float score to Sentiment
                if res.sentiment_score > 0.1:
                    sentiment = Sentiment.POSITIVE
                elif res.sentiment_score < -0.1:
                    sentiment = Sentiment.NEGATIVE
                else:
                    sentiment = Sentiment.NEUTRAL
                source_agent = f"Fusion_{res.id[:4]}"
                action = "MAINTAIN" # FusionResult usually doesn't carry override flags
                
            # --- Standard SupervisionResult ---
            elif isinstance(res, SupervisionResult):
                confidence = res.decision.confidence or 0.5
                sentiment = res.decision.final_sentiment
                source_agent = res.source_agent
                action = res.decision.action
                if action == "OVERRIDE":
                    override_reasons.append(f"({source_agent}: {res.decision.reason})")
                final_action_flags.add(action)
            else:
                logger.warning(f"Voter received unknown type: {type(res)}. Skipping.")
                continue
            
            # --- Bayesian Update ---
            prob_positive = 0.5 # NEUTRAL 的情况
            if sentiment == Sentiment.POSITIVE:
                prob_positive = confidence
            elif sentiment == Sentiment.NEGATIVE:
                prob_positive = 1.0 - confidence
            
            signal_log_odds = self._to_log_odds(prob_positive)
            total_log_odds += signal_log_odds
            total_confidence_weight += confidence
            
            logger.debug(f"  - Decision {source_agent}: Sent={sentiment}, Conf={confidence:.2f} -> P(Pos)={prob_positive:.2f} -> LogOdds={signal_log_odds:.2f}")

        # --- 融合结果 ---
        
        # 将总对数几率转换回最终概率
        final_prob_positive = self._from_log_odds(total_log_odds)
        
        # 确定最终情感
        neutral_low = 0.5 - (self.neutral_threshold / 2.0)
        neutral_high = 0.5 + (self.neutral_threshold / 2.0)
        
        if final_prob_positive < neutral_low:
            final_sentiment = Sentiment.NEGATIVE
            # 置信度是从 0.5 向下映射到 0
            final_confidence = (neutral_low - final_prob_positive) / neutral_low
        elif final_prob_positive > neutral_high:
            final_sentiment = Sentiment.POSITIVE
            # 置信度是从 0.5 向上映射到 1
            final_confidence = (final_prob_positive - neutral_high) / (1.0 - neutral_high)
        else:
            final_sentiment = Sentiment.NEUTRAL
            final_confidence = 0.0 # 中性区域的置信度为 0

        # 确保置信度在 [0, 1] 范围内
        final_confidence = max(0.0, min(1.0, final_confidence))

        # 确定最终行动
        final_action = "MAINTAIN"
        if "OVERRIDE" in final_action_flags or override_reasons:
             final_action = "OVERRIDE"
        elif "FLAG_FOR_REVIEW" in final_action_flags:
             final_action = "FLAG_FOR_REVIEW"

        reason = (
            f"Bayesian fusion of {len(decisions)} decisions. "
            f"Final P(Positive)={final_prob_positive:.3f}. "
            f"Result: {final_sentiment.value} (Conf={final_confidence:.3f}). "
            f"Action '{final_action}' triggered."
        )
        if override_reasons:
            reason += " Override reasons: " + " | ".join(override_reasons)

        logger.info(reason)

        return SupervisionResult(
            source_agent="Voter",
            decision=Decision(
                action=final_action,
                final_sentiment=final_sentiment,
                confidence=final_confidence,
                reason=reason
            ),
            timestamp=asyncio.get_event_loop().time()
        )

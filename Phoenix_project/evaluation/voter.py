import math
from typing import List
import asyncio

from ..core.schemas.supervision_result import SupervisionResult, Decision
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

    async def vote(self, decisions: List[SupervisionResult]) -> SupervisionResult:
        """
        使用贝叶斯融合逻辑聚合 L2 决策。

        Args:
            decisions: L2 智能体的 SupervisionResult 列表。

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

        for res in decisions:
            decision = res.decision
            
            # 将情感和置信度转换为 P(Positive)
            # 我们假设置信度代表了 P(AssignedSentiment)
            confidence = decision.confidence or 0.5
            
            prob_positive = 0.5 # NEUTRAL 的情况
            if decision.final_sentiment == Sentiment.POSITIVE:
                prob_positive = confidence
            elif decision.final_sentiment == Sentiment.NEGATIVE:
                prob_positive = 1.0 - confidence
            
            # 将 P(Positive) 转换为对数几率
            # 我们使用置信度本身作为权重，来调整弱信号的影响
            # (例如，一个 0.5 置信度的信号不应该贡献任何对数几率)
            # weight = (confidence - 0.5) * 2  (映射 [0.5, 1] -> [0, 1])
            # 
            # 简化：我们直接使用 logit，但我们会根据原始置信度对其进行加权
            # 一个更简单的方法：
            # (P(Pos)=0.8) -> log(0.8/0.2) = 1.38
            # (P(Pos)=0.6) -> log(0.6/0.4) = 0.40
            # (P(Pos)=0.5) -> log(0.5/0.5) = 0.0
            
            signal_log_odds = self._to_log_odds(prob_positive)
            
            # 按置信度加权信号 (一个 100% 置信度的信号贡献全部 log_odds)
            # (一个 50% 置信度的信号 [即 P(Pos)=0.5] 贡献 0 log_odds)
            # (一个 60% 置信度的信号 [P(Pos)=0.6] 贡献 0.2 * log(0.6/0.4)) -> 不，这太复杂了
            
            # 让我们保持简单：每个信号都贡献其全部的对数几率
            total_log_odds += signal_log_odds
            total_confidence_weight += confidence # 跟踪总置信度以用于归一化
            
            logger.debug(f"  - Decision {res.source_agent}: Sent={decision.final_sentiment}, Conf={confidence:.2f} -> P(Pos)={prob_positive:.2f} -> LogOdds={signal_log_odds:.2f}")

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

        # 确定最终行动 (如果任何一个 L2 智能体要求否决，则升级为否决)
        final_action = "MAINTAIN"
        override_reasons = []
        for res in decisions:
            if res.decision.action == "OVERRIDE":
                final_action = "OVERRIDE"
                override_reasons.append(f"({res.source_agent}: {res.decision.reason})")
            elif res.decision.action == "FLAG_FOR_REVIEW" and final_action != "OVERRIDE":
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

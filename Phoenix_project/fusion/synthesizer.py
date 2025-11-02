"""
Synthesizer Module.

Fuses multiple AgentDecisions into a single, robust FusionResult.
It uses a combination of algorithmic methods (e.g., Bayesian averaging)
and a MetacognitiveAgent (LLM) to generate the final output.
"""
import logging
import numpy as np
from typing import List, Dict, Any, Optional

# 修复：将相对导入 'from ..ai.metacognitive_agent...' 更改为绝对导入
from ai.metacognitive_agent import MetacognitiveAgent
# 修复：将相对导入 'from ..core.schemas.fusion_result...' 更改为绝对导入
from core.schemas.fusion_result import AgentDecision, FusionResult
# 修复：将相对导入 'from ..core.schemas.data_schema...' 更改为绝对导入
from core.schemas.data_schema import MarketEvent

logger = logging.getLogger(__name__)

class Synthesizer:
    """
    Fuses a list of AgentDecisions into a single FusionResult.
    """

    def __init__(self, metacognitive_agent: Optional[MetacognitiveAgent] = None):
        """
        Initializes the Synthesizer.

        Args:
            metacognitive_agent: An optional LLM agent used for
                                 generating a qualitative rationale.
        """
        self.metacognitive_agent = metacognitive_agent
        logger.info("Synthesizer initialized.")

    def _calculate_cognitive_uncertainty(self, decisions: List[AgentDecision]) -> float:
        """
        Calculates a measure of disagreement (uncertainty) among agents.
        A simple approach is the variance of their sentiment scores.
        """
        if len(decisions) < 2:
            return 0.0  # No disagreement if 0 or 1 agent

        sentiments = [d.sentiment for d in decisions]
        variance = np.var(sentiments)
        
        # Normalize variance. Max variance for scores in [-1, 1] is 1.0
        # (e.g., half at -1, half at 1).
        normalized_uncertainty = np.clip(variance, 0.0, 1.0)
        return float(normalized_uncertainty)

    def _fuse_scores_bayesian(self, decisions: List[AgentDecision]) -> Dict[str, float]:
        """
        Fuses scores using a confidence-weighted average (Bayesian-inspired).
        
        - Agents with higher confidence get a stronger "vote".
        - This prevents a low-confidence "gamble" from skewing the result.
        """
        if not decisions:
            return {"sentiment": 0.0, "impact": 0.0, "confidence": 0.0}

        total_confidence = sum(d.confidence for d in decisions)
        
        if total_confidence == 0:
            # If all agents are 0 confidence, result is neutral 0 confidence
            return {"sentiment": 0.0, "impact": 0.0, "confidence": 0.0}

        # Weighted average for sentiment and impact
        fused_sentiment = sum(d.sentiment * d.confidence for d in decisions) / total_confidence
        fused_impact = sum(d.predicted_impact * d.confidence for d in decisions) / total_confidence
        
        # Fused confidence could be the max, average, or a more complex model.
        # Using the average confidence of the "winning" side is robust.
        # For simplicity here, we use the weighted average confidence.
        fused_confidence = total_confidence / len(decisions) # Simple average
        
        return {
            "sentiment": float(fused_sentiment), 
            "impact": float(fused_impact), 
            "confidence": float(fused_confidence)
        }

    async def synthesize(self, 
                         event: MarketEvent,
                         context: Dict[str, Any], 
                         decisions: List[AgentDecision]) -> Optional[FusionResult]:
        """
        Orchestrates the fusion process.

        Args:
            event: The original MarketEvent being analyzed.
            context: The full RAG evidence context.
            decisions: The list of AgentDecision objects from the ensemble.

        Returns:
            A single FusionResult object, or None if fusion fails.
        """
        if not decisions:
            logger.warning("Cannot synthesize: No agent decisions provided.")
            return None

        try:
            # 1. Algorithmic Fusion (Quantitative)
            fused_scores = self._fuse_scores_bayesian(decisions)
            cognitive_uncertainty = self._calculate_cognitive_uncertainty(decisions)
            
            # 2. Metacognitive Fusion (Qualitative Rationale)
            fused_rationale = "Algorithmic fusion complete."
            if self.metacognitive_agent:
                event_context = {"event": event.dict(), "context": context}
                meta_result = await self.metacognitive_agent.synthesize(event_context, decisions)
                
                if meta_result and 'fused_rationale' in meta_result:
                    fused_rationale = meta_result['fused_rationale']
                    # Optional: Allow the meta-agent to override scores
                    # fused_scores['sentiment'] = meta_result.get('fused_sentiment', fused_scores['sentiment'])
                    # ...
                else:
                    logger.warning("Metacognitive agent failed to provide a rationale.")
            
            # 3. Assemble the final result
            # 修复：使用 FusionResult schema
            fusion_result = FusionResult(
                fused_confidence=fused_scores['confidence'],
                fused_sentiment=fused_scores['sentiment'],
                fused_predicted_impact=fused_scores['impact'],
                cognitive_uncertainty=cognitive_uncertainty,
                fused_rationale=fused_rationale,
                contributing_decisions=decisions
            )
            
            logger.info(f"Synthesis complete. Uncertainty: {cognitive_uncertainty:.3f}, Sentiment: {fused_scores['sentiment']:.3f}")
            return fusion_result

        except Exception as e:
            logger.error(f"Error during synthesis: {e}", exc_info=True)
            return None

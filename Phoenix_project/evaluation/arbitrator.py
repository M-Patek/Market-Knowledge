"""
Arbitrator Module.

This component is responsible for resolving conflicts when the Synthesizer
reports high cognitive uncertainty. It can trigger a secondary, more
focused cognitive loop.
"""
import logging
from typing import List, Optional

# 修复：使用正确的相对导入
from ..core.schemas.fusion_result import FusionResult, AgentDecision
from ..ai.metacognitive_agent import MetacognitiveAgent
from ..ai.ensemble_client import EnsembleClient

logger = logging.getLogger(__name__)

class Arbitrator:
    """
    Handles high-uncertainty scenarios by invoking a meta-cognitive agent
    or a specialized "Arbitrator" agent to make a final decision.
    """

    def __init__(self, 
                 uncertainty_threshold: float,
                 meta_agent: MetacognitiveAgent):
        """
        Initializes the Arbitrator.

        Args:
            uncertainty_threshold: The cognitive uncertainty level (0.0 to 1.0)
                                   above which arbitration is triggered.
            meta_agent: A strong "Arbitrator" LLM agent, passed as a
                        MetacognitiveAgent instance, to resolve the conflict.
        """
        self.uncertainty_threshold = uncertainty_threshold
        self.meta_agent = meta_agent
        if not self.meta_agent:
            raise ValueError("Arbitrator requires a MetacognitiveAgent to function.")
            
        logger.info(f"Arbitrator initialized with threshold: {self.uncertainty_threshold}")

    async def needs_arbitration(self, fusion_result: FusionResult) -> bool:
        """
        Checks if the given fusion result exceeds the uncertainty threshold.
        """
        needs_arbitration = fusion_result.cognitive_uncertainty > self.uncertainty_threshold
        if needs_arbitration:
            logger.warning(f"Arbitration needed: Uncertainty {fusion_result.cognitive_uncertainty:.3f} "
                           f"> threshold {self.uncertainty_threshold:.3f}")
        return needs_arbitration

    async def resolve_conflict(self, 
                               event_context: dict, 
                               conflicting_decisions: List[AgentDecision]) -> Optional[dict]:
        """
        Invokes the MetacognitiveAgent to analyze the conflicting decisions
        and produce a final, overriding judgment.

        Args:
            event_context: The original event and RAG context.
            conflicting_decisions: The list of decisions that caused the conflict.

        Returns:
            A dictionary containing the arbitrator's final decision, or None.
        """
        logger.info(f"Resolving conflict for {len(conflicting_decisions)} decisions...")
        
        try:
            # The MetacognitiveAgent's "synthesize" method is used here
            # with a prompt designed for arbitration (e.g., "arbitrator.json").
            arbitrated_result = await self.meta_agent.synthesize(
                event_context=event_context,
                agent_decisions=conflicting_decisions
            )
            
            if arbitrated_result:
                logger.info("Conflict resolved by Arbitrator.")
                # This result is expected to contain the fields
                # 'fused_rationale', 'fused_sentiment', etc.
                return arbitrated_result
            else:
                logger.error("Arbitrator meta-agent failed to produce a result.")
                return None
                
        except Exception as e:
            logger.error(f"Error during conflict resolution: {e}", exc_info=True)
            return None

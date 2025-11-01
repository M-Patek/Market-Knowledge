import numpy as np
from typing import Dict, Any, List, Tuple

from ..monitor.logging import get_logger
from ..core.schemas.fusion_result import AgentDecision

logger = get_logger(__name__)

class BayesianSynthesizer:
    """
    Fuses multiple, potentially conflicting, agent decisions into a
    single, probabilistic final decision.
    
    This implementation uses a simple weighted average as a placeholder.
    A true Bayesian implementation would:
    1. Define priors for each agent (their historical accuracy).
    2. Define likelihoods (e.g., P(Agent_Decision | True_State)).
    3. Compute the posterior: P(True_State | Agent_Decisions)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the Synthesizer.
        
        Args:
            config: Main system configuration.
        """
        self.config = config.get('synthesizer', {})
        # Agent priors (historical accuracy) - Placeholder
        self.agent_priors = {
            "analyst": 0.7,
            "fact_checker": 0.8,
            "context_observer": 0.65,
            "default": 0.6
        }
        logger.info("BayesianSynthesizer initialized (using weighted average).")

    def fuse(
        self,
        decisions: List[AgentDecision],
        source_credibility: Dict[str, float] = None
    ) -> Tuple[AgentDecision, Dict[str, float]]:
        """
        Fuses a list of agent decisions into a final decision and uncertainty score.
        
        Args:
            decisions (List[AgentDecision]): The list of outputs from the agent ensemble.
            source_credibility (Dict, optional): Scores for the RAG sources (not used
                                                 in this simple implementation).
                                                 
        Returns:
            A tuple of (AgentDecision, Dict[str, float]):
            1. The final, synthesized AgentDecision.
            2. A dictionary of uncertainty metrics.
        """
        
        # Filter out any error/invalid responses
        valid_decisions = [
            d for d in decisions 
            if d.decision not in ["ERROR", "INVALID_RESPONSE", "UNKNOWN"]
        ]
        
        if not valid_decisions:
            logger.warning("Synthesizer received no valid decisions. Defaulting to HOLD.")
            return self._default_decision(), self._default_uncertainty()

        # --- Simple Weighted Average Implementation ---
        
        total_weight = 0
        weighted_score = 0
        justifications = []
        
        # Map qualitative decisions to a numerical score
        decision_map = {"BUY": 1, "HOLD": 0, "SELL": -1}
        
        for d in valid_decisions:
            if d.decision not in decision_map:
                continue
                
            # Weight = Agent_Prior * Agent_Confidence
            agent_prior = self.agent_priors.get(d.agent_id, self.agent_priors['default'])
            weight = agent_prior * d.confidence
            
            total_weight += weight
            weighted_score += weight * decision_map[d.decision]
            
            justifications.append(
                f"[{d.agent_id} (Prior: {agent_prior:.0%}, Conf: {d.confidence:.0%})]: "
                f"{d.justification}"
            )
        
        if total_weight == 0:
            logger.warning("Synthesizer had zero total weight. Defaulting to HOLD.")
            return self._default_decision(), self._default_uncertainty()
            
        # --- Final Decision ---
        final_numeric_score = weighted_score / total_weight
        
        # Discretize the final score back into BUY/HOLD/SELL
        final_decision_str = self._discretize_score(final_numeric_score)
        
        # --- Uncertainty Calculation ---
        
        # 1. Confidence: The average *weighted* confidence
        avg_confidence = np.mean([d.confidence for d in valid_decisions])
        
        # 2. Agreement (Variance)
        # Calculate the variance of the numerical decisions
        decision_scores = [decision_map[d.decision] for d in valid_decisions]
        agreement_score = 1.0 - np.var(decision_scores) # 1.0 = full agreement, 0.0 = max disagreement (BUY/SELL)
        
        # 3. Final Cognitive Uncertainty
        # A simple model: Uncertainty = 1 - (Agreement * Confidence)
        cognitive_uncertainty = 1.0 - (agreement_score * avg_confidence)
        
        
        final_decision = AgentDecision(
            agent_id="synthesizer",
            decision=final_decision_str,
            confidence=avg_confidence, # Final confidence is the average
            justification="Fused Decision: " + " | ".join(justifications),
            metadata={
                "fused_numeric_score": final_numeric_score,
                "agreement_score": agreement_score
            }
        )
        
        uncertainty_metrics = {
            "cognitive_uncertainty": cognitive_uncertainty,
            "agreement": agreement_score,
            "avg_confidence": avg_confidence
        }

        return final_decision, uncertainty_metrics

    def _discretize_score(self, score: float) -> str:
        """Converts a numerical score (-1 to 1) to a decision string."""
        threshold = self.config.get('decision_threshold', 0.3)
        if score > threshold:
            return "BUY"
        elif score < -threshold:
            return "SELL"
        else:
            return "HOLD"

    def _default_decision(self) -> AgentDecision:
        """Returns a neutral default decision."""
        return AgentDecision(
            agent_id="synthesizer",
            decision="HOLD",
            confidence=0.0,
            justification="No valid agent inputs to synthesize."
        )

    def _default_uncertainty(self) -> Dict[str, float]:
        """Returns maximum uncertainty."""
        return {
            "cognitive_uncertainty": 1.0,
            "agreement": 0.0,
            "avg_confidence": 0.0
        }

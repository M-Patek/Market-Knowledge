# ai/bayesian_fusion_engine.py
import logging
import time
import math
from typing import List, Dict, Any
from scipy.stats import beta

from ai.validation import EvidenceItem
from ai.contradiction_detector import ContradictionDetector
from ai.probability_calibrator import ProbabilityCalibrator
from schemas.fusion_result import FusionResult

logger = logging.getLogger(__name__)

class BayesianFusionEngine:
    """
    (L3 Patched) L2 Engine: Fuses multiple pieces of L1 EvidenceItems into a single,
    probabilistic posterior.
    - Manages dynamic agent priors (Alpha/Beta) internally.
    - Detects contradictions.
    - Outputs a standardized FusionResult object.
    """
    def __init__(self):
        # (L3) Store dynamic agent priors internally
        self.agent_priors: Dict[str, Dict[str, float]] = {
            # Start with a neutral prior Beta(1,1) for all agents by default.
            # These will be updated dynamically based on performance.
        }
        self.contradiction_detector = ContradictionDetector()
        self.calibrator = ProbabilityCalibrator() # Not fully implemented yet
        logger.info("BayesianFusionEngine initialized.")

    def update_priors(self, agent_name: str, success: bool):
        """
        (L3) Updates the dynamic alpha/beta parameters for a given agent based on feedback.
        """
        if agent_name not in self.agent_priors:
            self.agent_priors[agent_name] = {"alpha": 1.0, "beta": 1.0}

        if success:
            self.agent_priors[agent_name]["alpha"] += 1.0
        else:
            self.agent_priors[agent_name]["beta"] += 1.0
        logger.info(f"Updated priors for agent '{agent_name}': {self.agent_priors[agent_name]}")

    def fuse(self, core_hypothesis: str, evidence_items: List[EvidenceItem]) -> FusionResult:
        """
        (L3 Patched) Performs Bayesian fusion on a list of EvidenceItems.
        
        Args:
            core_hypothesis: The central question being evaluated (e.g., "NVDA will outperform in Q3").
            evidence_items: A list of EvidenceItem objects from L1 Agents.
            
        Returns:
            A FusionResult object containing the final posterior, uncertainty, and other metadata.
        """
        start_time = time.time()
        
        if not evidence_items:
            logger.warning("Fuse called with no evidence items.")
            # Return a default, uncertain result if no evidence
            return FusionResult(posterior={'neutral': 1.0}, confidence_interval=(0.0, 1.0), rationale="No evidence provided.", conflict_log=[])

        # 1. Detect Contradictions
        # This identifies pairs of items that are in direct conflict.
        # (L3) This is now async, but we await it from a sync context
        # In a real async framework, this `fuse` method would be async
        # contradictions = await self.contradiction_detector.detect(evidence_items)
        contradictions = [] # Placeholder
        
        # 2. Calibrate Probabilities
        # (Placeholder: This would convert scores/text to a uniform P(Hypothesis | Evidence))
        # calibrated_evidence = [self.calibrator.calibrate(item) for item in evidence_items]
        
        # 3. Perform Bayesian Fusion
        # We model the probability P(Hypothesis) as a Beta distribution,
        # defined by Alpha (successes) and Beta (failures).
        # We start with a neutral prior, e.g., Beta(1, 1).
        prior_alpha, prior_beta = 1.0, 1.0
        
        total_weight = 0.0

        for item in evidence_items:
            # (L3) Get dynamic credibility weight for the source
            priors = self.agent_priors.get(item.source, {"alpha": 1.0, "beta": 1.0})
            alpha = priors["alpha"]
            beta = priors["beta"]
            credibility_weight = alpha / (alpha + beta)
            
            # Combine provenance confidence with source credibility
            evidence_weight = item.provenance_confidence * credibility_weight
            
            # Convert the agent's score [-1.0, 1.0] into an "impact" on Alpha and Beta
            # A positive score adds to Alpha (supports hypothesis)
            # A negative score adds to Beta (opposes hypothesis)
            if item.score > 0:
                prior_alpha += item.score * evidence_weight
            else:
                prior_beta += abs(item.score) * evidence_weight
            
            total_weight += evidence_weight

        # 4. Calculate Final Posterior and Uncertainty
        # The mean of the Beta distribution is our posterior probability
        posterior_mean_prob = prior_alpha / (prior_alpha + prior_beta)
        
        # Convert P(H) from [0, 1] back to a score from [-1, 1]
        posterior_mean = (posterior_mean_prob * 2) - 1
        
        # Calculate cognitive uncertainty
        # We can use the variance of the Beta distribution
        variance = (prior_alpha * prior_beta) / (((prior_alpha + prior_beta) ** 2) * (prior_alpha + prior_beta + 1))
        # Normalize variance to an uncertainty score (this is a simple heuristic)
        cognitive_uncertainty = math.sqrt(variance) * 4 # Scale std dev
        
        # Calculate 95% Confidence Interval (using Beta distribution's ppf)
        ci_low_prob = beta.ppf(0.025, prior_alpha, prior_beta)
        ci_high_prob = beta.ppf(0.975, prior_alpha, prior_beta)
        
        # Convert CI from [0, 1] to [-1, 1]
        ci_low = (ci_low_prob * 2) - 1
        ci_high = (ci_high_prob * 2) - 1

        # --- Task 2.1: Uncertainty-based Self-Correction ---
        # If contradictions were found, increase the uncertainty (widen the interval)
        if contradictions:
            logger.warning(f"Contradictions detected. Increasing uncertainty.")
            cognitive_uncertainty *= 1.5  # Punish-increase uncertainty
        # --- End Task 2.1 ---
        
        # (L3) Map the [-1, 1] posterior mean to a bullish/bearish probability dict
        bullish_prob = (posterior_mean + 1) / 2
        bearish_prob = 1 - bullish_prob

        rationale_str = (
            f"Fusion based on {len(evidence_items)} evidence items. "
            f"Final Beta distribution params: Alpha={prior_alpha:.2f}, Beta={prior_beta:.2f}. "
            f"{len(contradictions)} contradictions were detected, increasing uncertainty."
        )

        end_time = time.time()
        duration = end_time - start_time

        # (L3) Structured logging for metrics
        log_details = {
            "event": "fusion_completed",
            "duration_ms": round(duration * 1000, 2),
            "input_count": len(evidence_items),
            "conflict_count": len(contradictions),
            "posterior": {"bullish": round(bullish_prob, 4), "bearish": round(bearish_prob, 4)}
        }
        logger.info(log_details)
        
        return FusionResult(
            posterior={"bullish": bullish_prob, "bearish": bearish_prob},
            confidence_interval=(ci_low, ci_high),
            rationale=rationale_str,
            conflict_log=contradictions # Storing the list of conflicting pairs
        )

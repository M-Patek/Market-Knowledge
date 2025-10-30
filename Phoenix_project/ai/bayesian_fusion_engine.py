# ai/bayesian_fusion_engine.py
import logging
import math
from typing import List, Dict, Any
from scipy.stats import beta
import numpy as np

from models.evidence import EvidenceItem
from ai.contradiction_detector import ContradictionDetector
from ai.probability_calibrator import ProbabilityCalibrator
from ai.tabular_db_client import TabularDBClient # Placeholder

logger = logging.getLogger(__name__)

class SourceCredibilityStore:
    """
    Tracks the historical accuracy of L1 Agents to assign credibility weights.
    Uses a Beta distribution (Alpha, Beta) for each source.
    - Alpha: Success count (evidence was 'correct')
    - Beta: Failure count (evidence was 'incorrect')
    """
    def __init__(self, db_client: TabularDBClient):
        self.db_client = db_client
        # Load existing credibility from DB or start fresh
        self.store = self._load_credibility()
        logger.info(f"SourceCredibilityStore initialized with {len(self.store)} sources.")

    def _load_credibility(self) -> Dict[str, Dict[str, int]]:
        # Placeholder: Load from self.db_client
        # return self.db_client.get_all_records('source_credibility')
        return {
            "fundamental_analyst": {"alpha": 10, "beta": 2},
            "technical_analyst": {"alpha": 8, "beta": 3},
            "catalyst_monitor": {"alpha": 12, "beta": 1},
            "fact_checker_adversary": {"alpha": 5, "beta": 5}, # Neutral start
        }

    def update_credibility(self, source: str, success: bool):
        """
        Update the alpha or beta count for a source based on ground truth.
        """
        if source not in self.store:
            self.store[source] = {"alpha": 1, "beta": 1} # Start with Beta(1,1)
        
        if success:
            self.store[source]["alpha"] += 1
        else:
            self.store[source]["beta"] += 1
        
        # Placeholder: Save back to self.db_client
        # self.db_client.update_record('source_credibility', {'id': source}, self.store[source])
        logger.info(f"Updated credibility for {source}: {self.store[source]}")

    def get_credibility_params(self, source: str) -> (int, int):
        """Returns (alpha, beta) for a source."""
        return self.store.get(source, {"alpha": 1, "beta": 1}).values()

    def get_credibility_weight(self, source: str) -> float:
        """
        Calculates a simple credibility weight (mean of the Beta distribution).
        Weight = alpha / (alpha + beta)
        """
        params = self.store.get(source, {"alpha": 1, "beta": 1}) # Default to neutral prior
        alpha = params["alpha"]
        beta = params["beta"]
        return alpha / (alpha + beta)

class BayesianFusionEngine:
    """
    L2 Engine: Fuses multiple pieces of L1 EvidenceItems into a single,
    probabilistic judgment.
    """
    def __init__(self):
        # Placeholder: DB client should be injected
        self.db_client = TabularDBClient()
        self.credibility_store = SourceCredibilityStore(self.db_client)
        self.contradiction_detector = ContradictionDetector()
        self.calibrator = ProbabilityCalibrator() # Not fully implemented yet
        logger.info("BayesianFusionEngine initialized.")

    def fuse(self, core_hypothesis: str, evidence_items: List[EvidenceItem]) -> Dict[str, Any]:
        """
        Performs Bayesian fusion on a list of EvidenceItems.
        
        Args:
            core_hypothesis: The central question being answered (e.g., "Price of BTC will go up").
            evidence_items: A list of EvidenceItem objects from L1 Agents.
            
        Returns:
            A dictionary containing the final posterior, uncertainty, and other metadata.
        """
        if not evidence_items:
            logger.warning("Fuse called with no evidence items.")
            return {"error": "No evidence provided."}

        # 1. Detect Contradictions
        # This identifies pairs of items that are in direct conflict.
        contradictions = self.contradiction_detector.detect(evidence_items)
        if contradictions:
            logger.warning(f"ContradictionDetector flagged {len(contradictions)} pairs.")
        # --- Task 2.1: Strategy C Setup ---
        # Create a set of all items that are part of a contradiction for fast lookup
        contradicted_items_set = set()
        for item_pair in contradictions:
            contradicted_items_set.add(item_pair[0])
            contradicted_items_set.add(item_pair[1])

        # 2. Iterate list, update Bayesian Beta Distribution
        # Start with a neutral prior (e.g., Beta(1, 1) or Beta(2, 2))
        # A Beta(1,1) prior is a uniform distribution, representing max uncertainty.
        prior_alpha, prior_beta = 1.0, 1.0
        
        total_weight = 0.0

        for item in evidence_items:
            # Get credibility weight (mean of Beta dist.) for the source
            credibility_weight = self.credibility_store.get_credibility_weight(item.source)
            
            # Combine provenance confidence with source credibility
            evidence_weight = item.provenance_confidence * credibility_weight

            # --- Task 2.1: Strategy C (Automatic Weight Reduction) ---
            if item in contradicted_items_set:
                logger.warning(f"Applying contradiction penalty (x0.8) to item from {item.source} (Score: {item.score})")
                evidence_weight *= 0.8  # Punitive multiplier
            # --- End Task 2.1 ---
            
            # Map score [-1, 1] to alpha (success) and beta (failure) updates.
            # A high score (e.g., +1.0) adds to alpha.
            # A low score (e.g., -1.0) adds to beta.
            # A score of 0 adds to both, increasing uncertainty.
            update_strength = evidence_weight
            
            # Convert score to alpha/beta update
            alpha_update = max(0, item.score) * update_strength
            beta_update = max(0, -item.score) * update_strength
            
            prior_alpha += alpha_update
            prior_beta += beta_update
            total_weight += update_strength

        # 3. Calculate Final Posterior
        # The mean of the final Beta(alpha, beta) distribution
        posterior_mean_unscaled = prior_alpha / (prior_alpha + prior_beta)
        
        # Rescale from [0, 1] back to [-1, 1]
        posterior_mean = (posterior_mean_unscaled * 2) - 1
        
        dist = beta(prior_alpha, prior_beta)
        ci_low, ci_high = dist.interval(0.95)
        # "Cognitive Uncertainty" = width of the 95% confidence interval
        cognitive_uncertainty = ci_high - ci_low

        # --- Task 2.1: Strategy A (Increase Uncertainty) ---
        if contradictions:
            logger.warning(f"Applying Strategy A: Increasing cognitive uncertainty (x1.5) due to contradictions.")
            cognitive_uncertainty *= 1.5  # Punish-increase uncertainty
        # --- End Task 2.1 ---

        return {
            "final_posterior_mean": posterior_mean,
            "cognitive_uncertainty_score": cognitive_uncertainty,
            "final_alpha": prior_alpha,
            "final_beta": prior_beta,
            "total_evidence_weight": total_weight,
            "contradictions_detected": len(contradictions),
        }

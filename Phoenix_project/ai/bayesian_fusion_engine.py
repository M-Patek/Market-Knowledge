# ai/bayesian_fusion_engine.py
"""
Implements a probabilistic Evidence Fusion service with a Bayesian core.
This engine updates its beliefs based on incoming evidence using Bayes' theorem,
moving from simple heuristics to a mathematically rigorous framework.
"""
import logging
from typing import List, Dict, Any, Tuple
from scipy.stats import beta

from ai.validation import EvidenceItem
from .contradiction_detector import ContradictionDetector
from .embedding_client import EmbeddingClient

class SourceCredibilityStore:
    """
    Manages the dynamic credibility scores for various evidence sources.
    Each source's credibility is modeled as a Beta distribution.
    """
    def __init__(self, default_alpha: float = 2.0, default_beta: float = 8.0):
        """
        Initializes the store with a default credibility for unknown sources.
        A default of (2, 8) represents a prior belief that sources are more likely
        to be unreliable than reliable until proven otherwise.
        """
        self.logger = logging.getLogger("PhoenixProject.SourceCredibilityStore")
        self.default_prior = (default_alpha, default_beta)
        self.credibility_scores: Dict[str, Tuple[float, float]] = {
            # Pre-seed with some known high-quality sources
            "SEC EDGAR": (10.0, 2.0),
            "Reuters": (8.0, 3.0),
            "Bloomberg": (8.0, 3.0),
        }
        self.logger.info("SourceCredibilityStore initialized.")

    def get_credibility_prior(self, source: str) -> Tuple[float, float]:
        """Gets the current credibility parameters for a given source."""
        return self.credibility_scores.get(source, self.default_prior)

    def update_source_credibility(self, source: str, was_correct: bool):
        """
        Updates a source's credibility based on feedback.
        This would be called by a downstream validation process.
        """
        alpha, beta = self.get_credibility_prior(source)
        if was_correct:
            alpha += 1.0
        else:
            beta += 1.0
        self.credibility_scores[source] = (alpha, beta)
        self.logger.info(f"Updated credibility for source '{source}': New prior is Beta({alpha:.2f}, {beta:.2f}).")


class BayesianFusionEngine:
    """
    Fuses evidence using a Bayesian framework to produce a posterior probability distribution.
    """
    def __init__(self, embedding_client: EmbeddingClient, prior_alpha: float = 2.0, prior_beta: float = 2.0):
        """
        Initializes the engine with a prior belief.

        Args:
            embedding_client (EmbeddingClient): Client for generating embeddings needed for contradiction detection.
            prior_alpha (float): The alpha parameter of the initial Beta distribution.
                                 Represents initial "successes" or positive evidence.
            prior_beta (float): The beta parameter of the initial Beta distribution.
                                Represents initial "failures" or negative evidence.
        """
        self.logger = logging.getLogger("PhoenixProject.BayesianFusionEngine")
        if prior_alpha <= 0 or prior_beta <= 0:
            raise ValueError("Prior parameters (alpha, beta) must be positive.")
        self.base_prior_alpha = prior_alpha
        self.base_prior_beta = prior_beta
        # The engine now uses a credibility store and a contradiction detector
        self.credibility_store = SourceCredibilityStore()
        self.embedding_client = embedding_client
        self.contradiction_detector = ContradictionDetector(embedding_client)
        self.logger.info(f"BayesianFusionEngine initialized with prior Beta({prior_alpha}, {prior_beta}).")

    def _convert_evidence_to_likelihood(self, evidence: EvidenceItem) -> Tuple[float, float]:
        """
        Converts a single evidence item into likelihood counts (successes and failures).
        This is a simplified model where score and confidence directly influence the counts.
        """
        # A high score (e.g., 0.9) and high confidence (e.g., 0.8) should result in more "successes".
        # A low score (e.g., 0.2) should result in more "failures".
        # We can model successes as `confidence * score` and failures as `confidence * (1 - score)`.
        successes = evidence.provenance_confidence * evidence.score
        failures = evidence.provenance_confidence * (1.0 - evidence.score)
        return successes, failures

    def fuse(self, hypothesis: str, evidence_list: List[EvidenceItem]) -> Dict[str, Any]:
        """
        Updates the prior belief with a list of evidence to produce a posterior distribution.

        Args:
            hypothesis (str): A description of the hypothesis being tested.
            evidence_list (List[EvidenceItem]): A list of evidence items from the retriever.

        Returns:
            A dictionary containing the parameters and key stats of the posterior distribution.
        """
        self.logger.info(f"Fusing {len(evidence_list)} pieces of evidence for hypothesis: '{hypothesis}'")

        # --- [NEW] Adversarial Validation Step ---
        contradictions = self.contradiction_detector.detect(evidence_list)
        if contradictions:
            return {
                "hypothesis": hypothesis,
                "status": "LOW_CONSENSUS",
                "reason": "Contradictory evidence was detected.",
                "posterior_distribution": None,
                "summary_statistics": None,
                "contradictory_pairs": [[e1.dict(), e2.dict()] for e1, e2 in contradictions]
            }
        
        # Start with the base prior for the hypothesis itself
        posterior_alpha = self.base_prior_alpha
        posterior_beta = self.base_prior_beta

        # Update the belief iteratively for each piece of evidence
        total_successes = 0
        total_failures = 0
        for evidence in evidence_list:
            # [NEW] Get a dynamic prior based on the source's credibility
            source_prior_alpha, source_prior_beta = self.credibility_store.get_credibility_prior(evidence.source)

            successes, failures = self._convert_evidence_to_likelihood(evidence)
            # The update is now influenced by both the evidence itself AND the source's credibility
            posterior_alpha += successes * (source_prior_alpha / (source_prior_alpha + source_prior_beta))
            posterior_beta += failures
            total_successes += successes

        self.logger.info(f"Evidence update: Added {total_successes:.2f} successes and {total_failures:.2f} failures.")

        # The posterior distribution is a new Beta distribution
        posterior_dist = beta(posterior_alpha, posterior_beta)
        mean_prob = posterior_dist.mean()
        cred_interval = posterior_dist.interval(0.95) # 95% credible interval

        return {
            "hypothesis": hypothesis,
            "posterior_distribution": {
                "type": "Beta",
                "alpha": posterior_alpha,
                "beta": posterior_beta
            },
            "summary_statistics": {
                "mean_probability": mean_prob,
                "median_probability": posterior_dist.median(),
                "std_dev": posterior_dist.std(),
                "95_credible_interval": [cred_interval[0], cred_interval[1]]
            },
            "key_evidence": sorted(
                [e.dict() for e in evidence_list], 
                key=lambda x: x.get('provenance_confidence', 0) * x.get('score', 0), 
                reverse=True
            )[:5] # Return the top 5 most impactful pieces of evidence
        }

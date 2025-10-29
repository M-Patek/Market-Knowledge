import os
import json
import glob
import math
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from scipy.stats import beta
import numpy as np

# Internal dependencies (to be added)
from .contradiction_detector import ContradictionDetector # For Task 2.1
from .validation import EvidenceItem # For Task 2.1
from observability import get_logger

logger = get_logger(__name__)


# --- Task 2.2: Source Credibility Store ---

class SourceCredibilityStore:
    """
    Manages persistence and updates for L1 Agent credibility scores.
    Uses Beta distribution (alpha, beta) parameters for each source.
    """
    def __init__(self, store_path: str = "credibility_store.json", prompts_dir: str = "Phoenix_project/prompts"):
        self.store_path = store_path
        self.prompts_dir = prompts_dir
        self.store = self._load_store()
        self._initialize_store()

    def _load_store(self) -> Dict[str, Dict[str, int]]:
        """Loads the credibility store from a JSON file."""
        if os.path.exists(self.store_path):
            try:
                with open(self.store_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.error("Failed to decode credibility store. Reinitializing.")
                return {}
        return {}

    def _save_store(self):
        """Saves the credibility store to a JSON file."""
        with open(self.store_path, 'w') as f:
            json.dump(self.store, f, indent=2)

    def _initialize_store(self):
        """Task 2.2: On system startup, create a default entry for every Agent."""
        logger.info(f"Initializing Credibility Store. Checking agents in {self.prompts_dir}...")
        agent_files = glob.glob(os.path.join(self.prompts_dir, "*.json"))
        updated = False
        for f_path in agent_files:
            agent_name = os.path.basename(f_path).replace('.json', '')
            if agent_name not in self.store:
                logger.info(f"Adding new agent '{agent_name}' to store with neutral prior Beta(2, 2).")
                self.store[agent_name] = {"alpha": 2, "beta": 2} # Neutral prior
                updated = True
        if updated:
            self._save_store()

    def get_credibility_params(self, agent_source: str) -> Tuple[int, int]:
        """Gets the (alpha, beta) parameters for a given agent source."""
        params = self.store.get(agent_source, {"alpha": 2, "beta": 2}) # Default to neutral if somehow missed
        return params['alpha'], params['beta']

    def update_source_credibility(self, agent_source: str, was_correct: bool):
        """Task 2.2: Update credibility based on feedback (called by L3)."""
        alpha, beta = self.get_credibility_params(agent_source)
        if was_correct:
            self.store[agent_source]['alpha'] = alpha + 1
        else:
            self.store[agent_source]['beta'] = beta + 1
        logger.info(f"Updated credibility for '{agent_source}': {self.store[agent_source]}")
        self._save_store()


# --- Task 2.1: Bayesian Fusion Engine ---

class BayesianFusionEngine:
    """
    Performs Bayesian fusion of evidence scores.
    """
    def __init__(self):
        """Initializes the engine and its dependencies."""
        self.credibility_store = SourceCredibilityStore()
        self.contradiction_detector = ContradictionDetector()

    def fuse(self, core_hypothesis: str, evidence_items: List[EvidenceItem]) -> Dict[str, Any]:
        """
        Task 2.1: Fuses a list of EvidenceItems into a final judgment.
        """
        logger.info(f"Starting Bayesian fusion for hypothesis: '{core_hypothesis}'")

        # 1. Call ContradictionDetector
        contradictions = self.contradiction_detector.detect(evidence_items)
        if contradictions:
            logger.warning(f"ContradictionDetector flagged {len(contradictions)} pairs.")
            # TODO: Add logic to down-weight or handle contradictions

        # 2. Iterate list, update Bayesian Beta Distribution
        # Start with a neutral prior (e.g., Beta(1, 1) or Beta(2, 2))
        total_alpha = 1.0
        total_beta = 1.0
        
        for item in evidence_items:
            # 3. Integrate Credibility
            cred_alpha, cred_beta = self.credibility_store.get_credibility_params(item.source)
            # Credibility weight = mean of the agent's Beta distribution
            credibility_weight = cred_alpha / (cred_alpha + cred_beta) 

            # Combine provenance confidence with source credibility
            evidence_weight = item.provenance_confidence * credibility_weight

            # Map score [-1, 1] to alpha (success) and beta (failure) updates.
            # A high score (e.g., +1.0) adds to alpha.
            # A low score (e.g., -1.0) adds to beta.
            # A score of 0.0 adds to neither.
            # We use a scaling factor to make updates significant
            update_strength = 5.0 * evidence_weight 

            successes = max(0, item.score) * update_strength
            failures = max(0, -item.score) * update_strength

            total_alpha += successes
            total_beta += failures

        # 3. Output structured dictionary
        dist = beta(total_alpha, total_beta)
        posterior_mean = dist.mean()
        posterior_variance = dist.var()
        ci_low, ci_high = dist.interval(0.95)
        # "Cognitive Uncertainty" = width of the 95% confidence interval
        cognitive_uncertainty = ci_high - ci_low

        return {
            "final_posterior_mean": posterior_mean,
            "final_posterior_variance": posterior_variance,
            "confidence_interval_95": (ci_low, ci_high),
            "cognitive_uncertainty_score": cognitive_uncertainty,
            "contradictions_found": len(contradictions),
            "final_beta_params": (total_alpha, total_beta)
        }

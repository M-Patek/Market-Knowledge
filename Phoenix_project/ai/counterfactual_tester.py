# ai/counterfactual_tester.py
"""
Implements a service for running automated counterfactual or "what-if"
tests on the BayesianFusionEngine to understand its reasoning and sensitivities.
"""
import logging
import copy
from typing import List, Dict, Any

from .bayesian_fusion_engine import BayesianFusionEngine
from ai.validation import EvidenceItem

class CounterfactualTester:
    """
    Orchestrates a suite of "what-if" tests on the fusion engine.
    """
    def __init__(self, fusion_engine: BayesianFusionEngine):
        """
        Initializes the tester with an instance of the fusion engine.
        """
        self.logger = logging.getLogger("PhoenixProject.CounterfactualTester")
        self.fusion_engine = fusion_engine
        self.logger.info("CounterfactualTester initialized.")

    def run_test_suite(self, hypothesis: str, evidence_list: List[EvidenceItem]) -> Dict[str, Any]:
        """
        Runs a full suite of counterfactual tests on a given set of evidence.

        Args:
            hypothesis: The hypothesis being tested.
            evidence_list: The full list of evidence items.

        Returns:
            A structured report detailing the results of each test scenario.
        """
        if not evidence_list:
            self.logger.warning("Cannot run test suite: evidence list is empty.")
            return {}

        self.logger.info(f"Running counterfactual test suite for hypothesis: '{hypothesis}'")
        report = {"hypothesis": hypothesis, "scenarios": {}}

        # --- Baseline Scenario ---
        baseline_result = self.fusion_engine.fuse(hypothesis, evidence_list)
        baseline_prob = baseline_result.get("summary_statistics", {}).get("mean_probability")
        if baseline_prob is None:
             self.logger.error("Baseline fusion failed or resulted in low consensus. Halting tests.")
             report['baseline_result'] = baseline_result
             return report
        report['baseline_result'] = {"mean_probability": baseline_prob}


        # --- Scenario 1: Single Point of Failure Test ---
        # Find the most impactful piece of evidence (highest score * confidence)
        most_impactful_evidence = max(evidence_list, key=lambda e: e.score * e.provenance_confidence)
        evidence_without_strongest = [e for e in evidence_list if e != most_impactful_evidence]
        if evidence_without_strongest:
            spoof_result = self.fusion_engine.fuse(hypothesis, evidence_without_strongest)
            spoof_prob = spoof_result.get("summary_statistics", {}).get("mean_probability", baseline_prob)
            report['scenarios']['single_point_of_failure'] = {
                "description": "Result after removing the single most impactful piece of evidence.",
                "removed_evidence": most_impactful_evidence.finding,
                "new_mean_probability": spoof_prob,
                "sensitivity": baseline_prob - spoof_prob
            }

        # --- Scenario 2: Source-Type Ablation Test (Example: remove all 'news') ---
        evidence_without_news = [e for e in evidence_list if e.type != 'news']
        if evidence_without_news and len(evidence_without_news) < len(evidence_list):
            ablation_result = self.fusion_engine.fuse(hypothesis, evidence_without_news)
            ablation_prob = ablation_result.get("summary_statistics", {}).get("mean_probability", baseline_prob)
            report['scenarios']['source_ablation_news'] = {
                "description": "Result after removing all evidence of type 'news'.",
                "new_mean_probability": ablation_prob,
                "sensitivity": baseline_prob - ablation_prob
            }

        # --- Scenario 3: Evidence Inversion Test ---
        inverted_evidence_list = copy.deepcopy(evidence_list)
        # Find the most impactful positive evidence to invert
        positive_evidence = [e for e in inverted_evidence_list if e.score > 0.5]
        if positive_evidence:
            most_positive = max(positive_evidence, key=lambda e: e.score * e.provenance_confidence)
            most_positive.score = 1.0 - most_positive.score # Invert the score
            inversion_result = self.fusion_engine.fuse(hypothesis, inverted_evidence_list)
            inversion_prob = inversion_result.get("summary_statistics", {}).get("mean_probability", baseline_prob)
            report['scenarios']['evidence_inversion'] = {
                "description": "Result after inverting the score of the most impactful positive evidence.",
                "inverted_evidence": most_positive.finding,
                "new_mean_probability": inversion_prob,
                "sensitivity": baseline_prob - inversion_prob
            }

        self.logger.info("Counterfactual test suite complete.")
        return report

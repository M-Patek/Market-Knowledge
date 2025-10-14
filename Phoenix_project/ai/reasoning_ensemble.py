# ai/reasoning_ensemble.py
"""
Implements the multi-engine reasoning ensemble, the highest level of the
cognitive architecture. This module orchestrates various specialized
reasoners to produce a holistic, multi-faceted analysis.
"""
import logging
import numpy as np
import yaml
import asyncio
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from typing import Protocol, List, Dict, Any, NamedTuple
from pydantic import BaseModel

from ai.validation import EvidenceItem
from .bayesian_fusion_engine import BayesianFusionEngine

# --- 1. Standard Interface & Data Contracts ---

class ReasoningOutput(BaseModel):
    """A standardized output object for any reasoner."""
    reasoner_name: str
    conclusion: Any
    confidence: float
    supporting_evidence_ids: List[str]

class IReasoner(Protocol):
    """
    Defines the standard interface that all reasoning engines must adhere to.
    """
    async def reason(self, hypothesis: str, evidence: List[EvidenceItem]) -> ReasoningOutput:
        ...

# --- 2. Placeholder Implementations of Specialized Reasoners ---

class BayesianReasoner:
    """A wrapper for our existing BayesianFusionEngine to make it compliant with the IReasoner interface."""
    def __init__(self, fusion_engine: BayesianFusionEngine):
        self.engine = fusion_engine
        self.reasoner_name = "BayesianReasoner"

    async def reason(self, hypothesis: str, evidence: List[EvidenceItem]) -> ReasoningOutput:
        fusion_result = self.engine.fuse(hypothesis, evidence)
        stats = fusion_result.get("summary_statistics")
        
        if stats:
            return ReasoningOutput(
                reasoner_name=self.reasoner_name,
                conclusion={"posterior_mean_probability": stats["mean_probability"]},
                confidence=1.0 - stats["std_dev"], # Use std dev as an inverse proxy for confidence
                supporting_evidence_ids=[e.source_id for e in evidence if e.source_id]
            )
        else: # Handle contradiction case
            return ReasoningOutput(
                reasoner_name=self.reasoner_name,
                conclusion=fusion_result, # Pass the full "LOW_CONSENSUS" message
                confidence=0.0,
                supporting_evidence_ids=[]
            )

class SymbolicRuleReasoner:
    """A reasoner that applies hard-coded, deterministic domain knowledge."""
    def __init__(self):
        self.reasoner_name = "SymbolicRuleReasoner"
        self.rules = []
        self.logger = logging.getLogger(f"PhoenixProject.{self.reasoner_name}")
        try:
            with open("config/symbolic_rules.yaml", 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                self.rules = config.get('rules', [])
            self.logger.info(f"SymbolicRuleReasoner initialized with {len(self.rules)} rules.")
        except FileNotFoundError:
            self.logger.warning("config/symbolic_rules.yaml not found. SymbolicRuleReasoner will have no rules.")
        except Exception as e:
            self.logger.error(f"Failed to load symbolic rules: {e}")

    async def reason(self, hypothesis: str, evidence: List[EvidenceItem]) -> ReasoningOutput:
        for rule in self.rules:
            triggered, supporting_evidence = self._check_rule(rule, evidence)
            if triggered:
                self.logger.info(f"Rule '{rule.get('name')}' was triggered.")
                return ReasoningOutput(
                    reasoner_name=self.reasoner_name,
                    conclusion=rule.get('conclusion', {}),
                    confidence=rule.get('conclusion', {}).get('confidence', 1.0),
                    supporting_evidence_ids=[e.source_id for e in supporting_evidence if e.source_id]
                )
        
        # If no rules were triggered, return a default "no action" output.
        return ReasoningOutput(
            reasoner_name=self.reasoner_name,
            conclusion={"rule_triggered": "None"},
            confidence=1.0,
            supporting_evidence_ids=[]
        )

    def _check_rule(self, rule: Dict, evidence: List[EvidenceItem]) -> tuple[bool, List[EvidenceItem]]:
        """Checks if a single rule's conditions are met by the evidence."""
        conditions = rule.get('conditions', [])
        all_conditions_met = True
        supporting_evidence_for_rule = []

        for cond in conditions:
            filtered_evidence = [e for e in evidence if e.type == cond.get('evidence_type')]
            
            op_map = {
                "less_than": lambda s, t: s < t, "greater_than": lambda s, t: s > t,
                "less_than_or_equal_to": lambda s, t: s <= t, "greater_than_or_equal_to": lambda s, t: s >= t
            }
            
            score_op = op_map.get(cond.get('score_operator'))
            count_op = op_map.get(cond.get('count_operator'))

            if not score_op or not count_op: 
                all_conditions_met = False; break

            matching_evidence = [e for e in filtered_evidence if score_op(e.score, cond.get('score_threshold'))]
            
            if not count_op(len(matching_evidence), cond.get('count_threshold')):
                all_conditions_met = False; break
            
            supporting_evidence_for_rule.extend(matching_evidence)

        return all_conditions_met, list(set(supporting_evidence_for_rule))

class LLMExplainerReasoner:
    """A reasoner that uses a generative LLM to create a narrative explanation."""
    def __init__(self):
        self.reasoner_name = "LLMExplainerReasoner"
        # In a real system, this would be an actual LLM client.
        
    async def reason(self, hypothesis: str, evidence: List[EvidenceItem]) -> ReasoningOutput:
        # Placeholder Logic: Generate a simple summary.
        summary = f"Synthesized {len(evidence)} pieces of evidence regarding '{hypothesis}'. "
        summary += "Key findings include: " + "; ".join([e.finding for e in evidence[:2]])
        return ReasoningOutput(
            reasoner_name=self.reasoner_name,
            conclusion={"narrative_summary": summary},
            confidence=0.85, # Confidence is subjective for an LLM explainer
            supporting_evidence_ids=[e.source_id for e in evidence if e.source_id]
        )

class CausalInferenceReasoner:
    """A placeholder for a reasoner that would perform statistical causal tests."""
    def __init__(self):
        self.reasoner_name = "CausalInferenceReasoner"
        self.logger = logging.getLogger(f"PhoenixProject.{self.reasoner_name}")

    async def reason(self, hypothesis: str, evidence: List[EvidenceItem]) -> ReasoningOutput:
        # Find time-series evidence to compare. This is a simplified example.
        # A real implementation would need a more robust way to identify comparable time-series.
        ts_evidence = [e for e in evidence if e.type == 'market_data' and isinstance(e.finding, dict) and 'time_series' in e.finding]

        if len(ts_evidence) < 2:
            return ReasoningOutput(
                reasoner_name=self.reasoner_name,
                conclusion={"causal_finding": "Insufficient time-series evidence for causal analysis."},
                confidence=0.0,
                supporting_evidence_ids=[]
            )

        # Perform a Granger causality test between the first two time-series found.
        series_a = ts_evidence[0]
        series_b = ts_evidence[1]
        
        try:
            # Assume the time series are in a pandas DataFrame friendly format
            df = pd.DataFrame({
                series_a.source: series_a.finding['time_series'],
                series_b.source: series_b.finding['time_series']
            }).dropna()

            if len(df) < 20: # Need sufficient data for a meaningful test
                raise ValueError("Time series too short for Granger causality test.")

            # Test if series_b "Granger-causes" series_a
            max_lag = 5
            test_result = grangercausalitytests(df[[series_a.source, series_b.source]], max_lag, verbose=False)
            
            # Check the p-value of the F-test for the chosen lag
            p_value = test_result[max_lag][0]['ssr_ftest'][1]

            return ReasoningOutput(
                reasoner_name=self.reasoner_name,
                conclusion={"causal_finding": f"Granger causality test shows p-value of {p_value:.4f} for {series_b.source} predicting {series_a.source}.", "is_significant": p_value < 0.05},
                confidence=1.0 - p_value, # Use 1 minus p-value as a proxy for confidence
                supporting_evidence_ids=[s.source_id for s in [series_a, series_b] if s.source_id]
            )
        except Exception as e:
            self.logger.error(f"Causal inference test failed: {e}")
            return ReasoningOutput(
                reasoner_name=self.reasoner_name,
                conclusion={"causal_finding": f"Error during analysis: {e}"},
                confidence=0.0,
                supporting_evidence_ids=[]
            )


# --- 3. Meta-Learner for Synthesizing Outputs ---

class MetaLearner:
    """
    Learns the optimal weights for combining the outputs of multiple reasoners
    to produce a final, synthesized decision.
    """
    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
        from sklearn.svm import SVC

        self.logger = logging.getLogger("PhoenixProject.MetaLearner")
        
        # Define the 'AI Jury' - a diverse set of base models
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=10, random_state=42)),
            ('svc', SVC(probability=True, random_state=42))
        ]

        # The 'Chief Judge' (meta-model) learns from the jury's predictions.
        # We use a StackingClassifier to orchestrate this process.
        self.model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
        self.is_trained = False

    def _featurize(self, reasoner_outputs: List[ReasoningOutput]) -> np.ndarray:
        """Converts the diverse outputs of the ensemble into a flat feature vector."""
        features = []
        # A simple featurization: use the confidence score of each reasoner.
        # In a more advanced system, this would be much more sophisticated.
        for output in sorted(reasoner_outputs, key=lambda x: x.reasoner_name):
            features.append(output.confidence)
        return np.array(features).reshape(1, -1)

    def train(self, historical_outputs: List[List[ReasoningOutput]], true_outcomes: List[int]):
        """
        Trains the meta-learner on historical reasoner outputs and their true outcomes.
        """
        self.logger.info(f"Training MetaLearner on {len(historical_outputs)} historical data points...")
        X = np.vstack([self._featurize(outputs) for outputs in historical_outputs])
        y = np.array(true_outcomes)
        self.model.fit(X, y)
        self.is_trained = True
        self.logger.info("MetaLearner training complete.")
        # Log the learned weights (coefficients) for interpretability
        self.logger.info(f"Learned reasoner weights (coefficients): {self.model.final_estimator_.coef_}")

    def predict(self, reasoner_outputs: List[ReasoningOutput]) -> Dict[str, Any]:
        """
        Uses the trained model to produce a final, synthesized probability.
        """
        if not self.is_trained:
            # Fallback logic: if not trained, return an unweighted average of confidences.
            avg_confidence = np.mean([o.confidence for o in reasoner_outputs]) if reasoner_outputs else 0
            return {"final_probability": avg_confidence, "source": "unweighted_average"}

        features = self._featurize(reasoner_outputs)
        # Predict the probability of the positive class (1)
        final_prob = self.model。predict_proba(features)[0, 1]
        return {"final_probability": final_prob, "source": "meta_learner"}


# --- 4. The Main Ensemble Orchestrator ---

class ReasoningEnsemble:
    """
    Orchestrates a collection of specialized reasoners to run in parallel.
    """
    def __init__(self, *reasoners: IReasoner):
        """
        Initializes the ensemble with a list of reasoner instances.
        """
        self.logger = logging.getLogger("PhoenixProject.ReasoningEnsemble")
        self.reasoners = reasoners
        # [NEW] The ensemble now has a meta-learner
        self.meta_learner = MetaLearner()
        self.logger.info(f"ReasoningEnsemble initialized with {len(self.reasoners)} reasoners.")

    async def analyze(self, hypothesis: str, evidence: List[EvidenceItem]) -> Dict[str, Any]:
        """
        Runs all registered reasoners in parallel, synthesizes their outputs,
        and returns a final conclusion.
        """
        self.logger.info(f"Dispatching analysis to all {len(self.reasoners)} reasoners...")
        
        tasks = [
            reasoner.reason(hypothesis, evidence) for reasoner in self.reasoners
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        final_outputs = []
        for res in results:
            if isinstance(res, ReasoningOutput):
                final_outputs.append(res)
            elif isinstance(res, Exception):
                self.logger.error(f"A reasoner failed during execution: {res}")

        # --- [NEW] Meta-Learning Stage ---
        # Use the meta-learner to synthesize a final conclusion
        final_conclusion = self.meta_learner.predict(final_outputs)

        self.logger.info(f"Ensemble analysis complete. Final conclusion: {final_conclusion}")
        return {
            "final_conclusion": final_conclusion,
            "individual_reasoner_outputs": [o.dict() for o in final_outputs]
        }

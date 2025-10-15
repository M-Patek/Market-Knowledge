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
from .walk_forward_trainer import WalkForwardTrainer
from .contradiction_detector import ContradictionDetector
from .embedding_client import EmbeddingClient
from .probability_calibrator import ProbabilityCalibrator
import tensorflow as tf
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
            posterior_dist = fusion_result.get("posterior_distribution", {})
            return ReasoningOutput(
                reasoner_name=self.reasoner_name,
                conclusion={
                    "posterior_mean_probability": stats["mean_probability"],
                    # [NEW] Expose posterior and prior parameters for meta-feature calculation.
                    "posterior_alpha": posterior_dist.get("alpha"),
                    "posterior_beta": posterior_dist.get("beta"),
                    "prior_alpha": self.engine.base_prior_alpha,
                    "prior_beta": self.engine.base_prior_beta
                },
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
        from sklearn.ensemble import StackingClassifier
        from sklearn.svm import SVC
        # [NEW] Import deep learning components. This will require adding tensorflow to requirements.txt.
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Input
        # [REINFORCEMENT] Import state-of-the-art tree models.
        import xgboost as xgb
        import lightgbm as lgb

        self.logger = logging.getLogger("PhoenixProject.MetaLearner")
        
        # Define the 'AI Jury' - a diverse set of base models
        estimators = [
            ('xgb', xgb.XGBClassifier(n_estimators=10, random_state=42, use_label_encoder=False, eval_metric='logloss')),
            ('lgb', lgb.LGBMClassifier(n_estimators=10, random_state=42)),
            ('svc', SVC(probability=True, random_state=42))
        ]
        self.base_estimators = StackingClassifier(estimators=estimators, final_estimator=lgb.LGBMClassifier()) # Keep a simple final estimator for the base layer

        # [REPLACEMENT] The 'Chief Judge' is now a time-aware deep learning model.
        # It will learn from the sequence of predictions made by the base estimators.
        self.model = Sequential([
            # TODO: The input shape should be dynamically determined by the number of features.
            # For now, this is a placeholder. A more robust implementation would calculate this.
            Input(shape=(5, 3 * 3 + 4)), # (n_timesteps, n_reasoners * n_base_features + n_meta_features)
            LSTM(16, activation='relu'),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        # [NEW] Add the calibrator as the final step in the cognitive pipeline.
        self.calibrator = ProbabilityCalibrator(method='isotonic')
        # [NEW] Store the last known efficacies for prediction.
        self.last_known_efficacies = {}
        self.is_trained = False

    def _featurize(self, reasoner_output: ReasoningOutput) -> List[float]:
        """Converts a single reasoner's output into a feature vector."""
        features = []
        # Start with a base feature, the confidence score.
        features.append(reasoner_output.confidence)
        
        # [NEW] Extract deeper features based on reasoner type, per master's plan.
        if reasoner_output.reasoner_name == "BayesianReasoner" and isinstance(reasoner_output.conclusion, dict):
            prob = reasoner_output.conclusion.get("posterior_mean_probability", 0.5)
            features.append(prob) # Add the posterior probability
            features.append(abs(prob - 0.5) * 2) # Add a measure of conviction
        else:
            features.extend([0.5, 0.0]) # Add neutral features for other types

        return features

    def _create_sequences(self, features: np.ndarray, labels: np.ndarray, n_timesteps: int = 5) -> (np.ndarray, np.ndarray):
        """
        [NEW] Helper function to transform time-series data into sequences for the LSTM.
        """
        X, y = [], []
        for i in range(len(features) - n_timesteps + 1):
            X.append(features[i:(i + n_timesteps)])
            y.append(labels[i + n_timesteps - 1])
        return np.array(X), np.array(y)

    def train(
        self,
        historical_outputs: List[List[ReasoningOutput]],
        true_outcomes: List[int],
        contradiction_counts: List[int],
        historical_efficacies: List[Dict[str, float]]):
        """
        Trains the meta-learner on historical reasoner outputs and their true outcomes.
        [NEW] Includes an adversarial training loop for robustness.
        """
        self.logger.info(f"Training MetaLearner on {len(historical_outputs)} historical data points...")
        # 1. Featurize the full history
        # We create a flat feature matrix where each row is a day and columns are features from all reasoners.
        daily_features = []
        sorted_reasoner_names = sorted(list(historical_efficacies[0].keys())) if historical_efficacies else []
        
        for i, daily_outputs in enumerate(historical_outputs):
            day_feature_vector = []
            reasoner_probs = []
            sorted_outputs = sorted(daily_outputs, key=lambda x: x.reasoner_name)
            for output in sorted_outputs:
                day_feature_vector.extend(self._featurize(output))
                # Collect probabilities for dispersion calculation
                if output.reasoner_name == "BayesianReasoner" and isinstance(output.conclusion, dict):
                    reasoner_probs.append(output.conclusion.get("posterior_mean_probability", 0.5))
                else:
                    reasoner_probs.append(output.confidence) # Use confidence as a proxy

            # [NEW] Meta-Feature 1: Dispersion Penalty
            dispersion = np.std(reasoner_probs) if reasoner_probs else 0
            day_feature_vector.append(dispersion)
            
            # [NEW] Meta-Feature 2: Bayesian Prior Bias
            bayesian_out = next((o for o in daily_outputs if o.reasoner_name == "BayesianReasoner"), None)
            bayesian_bias = 0.0
            if bayesian_out and isinstance(bayesian_out.conclusion, dict) and bayesian_out.conclusion.get("posterior_alpha"):
                prior_ratio = bayesian_out.conclusion["prior_alpha"] / bayesian_out.conclusion["prior_beta"]
                posterior_ratio = bayesian_out.conclusion["posterior_alpha"] / bayesian_out.conclusion["posterior_beta"]
                bayesian_bias = (posterior_ratio / prior_ratio) - 1.0 if prior_ratio > 0 else 0.0
            day_feature_vector.append(bayesian_bias)
            
            # [NEW] Meta-Feature 3: Contradiction Count
            day_feature_vector.append(contradiction_counts[i])
            
            # [NEW] Meta-Feature 4: Reasoner Historical Efficacy
            efficacies = historical_efficacies[i]
            for name in sorted_reasoner_names:
                day_feature_vector.append(efficacies.get(name, 0.5))

            daily_features.append(day_feature_vector)
        
        X_flat = np.array(daily_features)
        y = np.array(true_outcomes)

        # 2. Create sequences
        n_timesteps = 5 # This should be configurable
        X_sequences, y_sequences = self._create_sequences(X_flat, y, n_timesteps)

        # 3. Train the LSTM model with an Adversarial Loop
        if X_sequences.shape[0] > 0:
            self.logger.info("Starting adversarial training loop...")
            # Convert to TensorFlow tensors
            dataset = tf.data.Dataset.from_tensor_slices((X_sequences, y_sequences)).batch(4)
            epsilon = 0.01 # Perturbation magnitude
            optimizer = tf.keras.optimizers.Adam()
            loss_fn = tf.keras.losses.BinaryCrossentropy()

            for epoch in range(20): # Number of epochs
                for step, (x_batch, y_batch) in enumerate(dataset):
                    with tf.GradientTape() as tape:
                        tape.watch(x_batch)
                        prediction = self.model(x_batch, training=True)
                        loss = loss_fn(y_batch, prediction)
                    # Get gradients of loss wrt the input
                    gradient = tape.gradient(loss, x_batch)
                    # Create the adversarial perturbation
                    perturbation = epsilon * tf.sign(gradient)
                    x_adversarial = x_batch + perturbation
                    # Train on the adversarial example
                    with tf.GradientTape() as inner_tape:
                        prediction = self.model(x_adversarial, training=True)
                        loss = loss_fn(y_batch, prediction)
                    grads = inner_tape.gradient(loss, self.model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            self.logger.info("Adversarial training complete. Now training probability calibrator...")

            # [NEW] 4. Train the ProbabilityCalibrator on the LSTM's predictions
            uncalibrated_probs = self.model.predict(X_sequences, verbose=0).flatten().tolist()
            self.calibrator.train(uncalibrated_probs, y_sequences.tolist())
        else:
            self.logger.warning("Not enough data to train MetaLearner LSTM or Calibrator.")

        self.is_trained = True

    def predict(
        self,
        reasoner_outputs: List[List[ReasoningOutput]],
        contradiction_count: int,
        reasoner_efficacies: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Uses the trained model to produce a final, synthesized probability.
        """
        if not self.is_trained:
            # Fallback logic: if not trained, return an unweighted average of confidences.
            avg_confidence = np.mean([o.confidence for o in reasoner_outputs[0]]) if reasoner_outputs and reasoner_outputs[0] else 0
            return {"final_probability": avg_confidence, "source": "unweighted_average"}
        
        # The 'predict' method now expects a sequence of recent outputs.
        # The caller (e.g., the backtest engine) would be responsible for passing this sequence.
        
        sorted_reasoner_names = sorted(list(reasoner_efficacies.keys()))
        daily_features_sequence = []
        for daily_outputs in reasoner_outputs: # Assuming reasoner_outputs is List[List[ReasoningOutput]]
            day_feature_vector = []
            reasoner_probs = []
            for output in sorted(daily_outputs, key=lambda x: x.reasoner_name):
                day_feature_vector.extend(self._featurize(output))
                if output.reasoner_name == "BayesianReasoner" and isinstance(output.conclusion, dict):
                    reasoner_probs.append(output.conclusion.get("posterior_mean_probability", 0.5))
                else:
                    reasoner_probs.append(output.confidence)
            
            # Calculate and append meta-features, same as in training.
            dispersion = np.std(reasoner_probs) if reasoner_probs else 0
            day_feature_vector.append(dispersion)
            
            bayesian_out = next((o for o in daily_outputs if o.reasoner_name == "BayesianReasoner"), None)
            bayesian_bias = 0.0
            if bayesian_out and isinstance(bayesian_out.conclusion, dict) and bayesian_out.conclusion.get("posterior_alpha"):
                prior_ratio = bayesian_out.conclusion["prior_alpha"] / bayesian_out.conclusion["prior_beta"]
                posterior_ratio = bayesian_out.conclusion["posterior_alpha"] / bayesian_out.conclusion["posterior_beta"]
                bayesian_bias = (posterior_ratio / prior_ratio) - 1.0 if prior_ratio > 0 else 0.0
            day_feature_vector.append(bayesian_bias)
            
            # [NEW] Meta-Feature 3: Contradiction Count
            day_feature_vector.append(contradiction_count)

            # [NEW] Meta-Feature 4: Reasoner Historical Efficacy
            # We use the same efficacy scores for each day in the short prediction sequence.
            for name in sorted_reasoner_names:
                day_feature_vector.append(reasoner_efficacies.get(name, 0.5))
            
            daily_features_sequence.append(day_feature_vector)

        features_3d = np.array(daily_features_sequence).reshape(1, len(daily_features_sequence), -1)
        uncalibrated_prob = self.model.predict(features_3d, verbose=0)[0][0]

        # [NEW] Apply the trained calibrator to the final output
        calibrated_prob = self.calibrator.calibrate([uncalibrated_prob])
        final_prob = calibrated_prob[0] if calibrated_prob else uncalibrated_prob

        return {"final_probability": final_prob, "source": "meta_learner"}


# --- 4. The Main Ensemble Orchestrator ---

class ReasoningEnsemble:
    """
    Orchestrates a collection of specialized reasoners to run in parallel.
    """
    def __init__(self, embedding_client: EmbeddingClient, *reasoners: IReasoner):
        """
        Initializes the ensemble with a list of reasoner instances.
        """
        # TODO: The calling script (phoenix_project.py) must be updated to pass the embedding_client.
        self.logger = logging.getLogger("PhoenixProject.ReasoningEnsemble")
        self.reasoners = reasoners
        # [NEW] The ensemble now has a meta-learner
        self.meta_learner = MetaLearner()
        # [NEW] The ensemble now has a contradiction detector to assess evidence quality.
        self.contradiction_detector = ContradictionDetector(embedding_client)
        self.logger.info(f"ReasoningEnsemble initialized with {len(self.reasoners)} reasoners.")

    def train(self, historical_data: List[Dict[str, Any]]):
        """
        [NEW] Centralizes the complex training process for the ensemble's meta-learner.
        """
        self.logger.info("Initiating training for the ReasoningEnsemble's MetaLearner...")
        reasoner_names = [r.reasoner_name for r in self.reasoners]
        trainer = WalkForwardTrainer(historical_data, reasoner_names)
        self.meta_learner = trainer.train(self.meta_learner)
        self.logger.info("MetaLearner training complete.")

    async def analyze(self, hypothesis: str, evidence: List[EvidenceItem]) -> Dict[str, Any]:
        """
        Runs all registered reasoners in parallel, synthesizes their outputs,
        and returns a final conclusion.
        """
        self.logger.info(f"Dispatching analysis to all {len(self.reasoners)} reasoners...")
        
        # [NEW] Step 1: Detect contradictions in the source evidence to use as a meta-feature.
        contradictions = self.contradiction_detector.detect(evidence)
        contradiction_count = len(contradictions)
        self.logger.info(f"Found {contradiction_count} contradictory evidence pairs.")
        
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
        # The historical sequence for prediction would need to be passed in a real scenario.
        # For now, we assume `final_outputs` represents the most recent data point in a sequence.
        final_conclusion = self.meta_learner.predict(
            [final_outputs],
            contradiction_count=contradiction_count,
            reasoner_efficacies=self.meta_learner.last_known_efficacies
        )

        self.logger.info(f"Ensemble analysis complete. Final conclusion: {final_conclusion}")
        return {
            "final_conclusion": final_conclusion,
            "individual_reasoner_outputs": [o.dict() for o in final_outputs]
        }

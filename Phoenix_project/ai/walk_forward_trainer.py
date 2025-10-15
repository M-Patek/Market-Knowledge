# ai/walk_forward_trainer.py
import logging
import numpy as np
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
from typing import List, Dict, Any
from .reasoning_ensemble import MetaLearner, ReasoningOutput

class WalkForwardTrainer:
    """
    Manages training a MetaLearner with dynamically calculated reasoner efficacy scores.
    """
    def __init__(self, historical_data: List[Dict[str, Any]], reasoner_names: List[str]):
        self.logger = logging.getLogger("PhoenixProject.WalkForwardTrainer")
        self.historical_data = historical_data
        self.reasoner_names = sorted(reasoner_names)

    def train(self, meta_learner: MetaLearner, n_splits: int = 5) -> MetaLearner:
        """
        Executes the walk-forward training process.
        In each fold, it calculates reasoner performance on the test set
        and uses it as a feature for training on the subsequent training set.
        """
        self.logger.info("Starting walk-forward training for MetaLearner...")
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Start with neutral efficacy scores
        last_efficacies = {name: 0.5 for name in self.reasoner_names}

        historical_outputs = [d['outputs'] for d in self.historical_data]
        true_outcomes = [d['outcome'] for d in self.historical_data]
        contradiction_counts = [d['contradiction_count'] for d in self.historical_data]

        for train_index, test_index in tscv.split(historical_outputs):
            self.logger.info(f"Training on fold with {len(train_index)} train samples and {len(test_index)} test samples.")
            
            if not train_index.size or not test_index.size:
                continue

            # 1. Prepare training data for this fold with efficacies from the *previous* fold
            train_outputs = [historical_outputs[i] for i in train_index]
            train_outcomes = [true_outcomes[i] for i in train_index]
            train_contradictions = [contradiction_counts[i] for i in train_index]
            efficacies_for_fold = [last_efficacies] * len(train_outputs)
            
            meta_learner.train(train_outputs, train_outcomes, train_contradictions, efficacies_for_fold)

            # 2. Calculate new efficacies on the test set to be used in the next iteration
            test_outputs = [historical_outputs[i] for i in test_index]
            test_outcomes_np = np.array([true_outcomes[i] for i in test_index])
            
            last_efficacies = self._calculate_reasoner_efficacy(test_outputs, test_outcomes_np)
            self.logger.info(f"Calculated new efficacies on test set: {last_efficacies}")

        self.logger.info("Walk-forward training complete. Storing final efficacy scores for prediction.")
        meta_learner.last_known_efficacies = last_efficacies
        return meta_learner

    def _calculate_reasoner_efficacy(self, daily_outputs_list: List[List[ReasoningOutput]], true_outcomes: np.ndarray) -> Dict[str, float]:
        efficacies = {}
        for name in self.reasoner_names:
            reasoner_preds = []
            for daily_outputs in daily_outputs_list:
                output = next((o for o in daily_outputs if o.reasoner_name == name), None)
                if output:
                    prob = 0.5
                    if isinstance(output.conclusion, dict) and "posterior_mean_probability" in output.conclusion:
                        prob = output.conclusion["posterior_mean_probability"]
                    else:
                        prob = output.confidence
                    reasoner_preds.append(prob)
                else:
                    reasoner_preds.append(0.5)
            
            brier = brier_score_loss(true_outcomes, reasoner_preds)
            efficacies[name] = 1.0 - brier
        return efficacies

# ai/probability_calibrator.py
"""
Implements a service for calibrating the probabilistic outputs of the
BayesianFusionEngine, ensuring they are statistically reliable.
"""
import logging
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from typing import List, Optional

from observability import PROBABILITY_CALIBRATION_BRIER_SCORE

class ProbabilityCalibrator:
    """
    Trains a calibration model and applies it to new predictions.
    """
    def __init__(self, method: str = 'isotonic'):
        """
        Initializes the calibrator.

        Args:
            method (str): The calibration method to use ('isotonic' or 'sigmoid').
                          'isotonic' is non-parametric and flexible, but requires more data.
                          'sigmoid' (Platt scaling) is parametric and works well for smaller datasets.
        """
        self.logger = logging.getLogger("PhoenixProject.ProbabilityCalibrator")
        if method not in ['isotonic', 'sigmoid']:
            raise ValueError("Method must be either 'isotonic' or 'sigmoid'.")
        
        self.method = method
        self.calibrator: Optional[CalibratedClassifierCV] = None
        self.logger.info(f"ProbabilityCalibrator initialized with method: '{self.method}'.")

    def train(self, uncalibrated_probs: List[float], true_outcomes: List[int]):
        """
        Trains the calibration model on historical predictions and their true outcomes.

        Args:
            uncalibrated_probs: A list of raw probabilities (e.g., from the Bayesian engine).
            true_outcomes: A list of the actual outcomes (1 for success, 0 for failure).
        """
        self.logger.info(f"Training calibration model on {len(uncalibrated_probs)} data points...")
        # scikit-learn's calibrator expects a shape of (n_samples, 1)
        X = np.array(uncalibrated_probs).reshape(-1, 1)
        y = np.array(true_outcomes)

        # We use a dummy classifier that does nothing, as we only need the calibration part.
        from sklearn.linear_model import LogisticRegression
        dummy_clf = LogisticRegression() 

        self.calibrator = CalibratedClassifierCV(
            estimator=dummy_clf,
            method=self.method,
            cv='prefit' # We are not fitting the dummy classifier
        )
        self.calibrator.fit(X, y)
        
        # After training, calculate and report the Brier score
        calibrated_probs = self.calibrate(uncalibrated_probs)
        if calibrated_probs:
            score = self.calculate_brier_score(calibrated_probs, true_outcomes)
            self.logger.info(f"Calibration complete. Brier Score Loss on training data: {score:.4f}")
            PROBABILITY_CALIBRATION_BRIER_SCORE.set(score)

    def calibrate(self, uncalibrated_probs: List[float]) -> Optional[List[float]]:
        """
        Applies the trained calibration model to a new set of probabilities.
        """
        if not self.calibrator:
            self.logger.warning("Calibrator has not been trained yet. Cannot calibrate probabilities.")
            return None
        
        X = np.array(uncalibrated_probs).reshape(-1, 1)
        # The output is an array of shape (n_samples, 2) with probabilities for class 0 and 1.
        # We want the probability of class 1 (success).
        calibrated_probs = self.calibrator.predict_proba(X)[:, 1]
        return calibrated_probs.tolist()

    def calculate_brier_score(self, calibrated_probs: List[float], true_outcomes: List[int]) -> float:
        """Calculates the Brier Score Loss, a measure of calibration accuracy."""
        return brier_score_loss(true_outcomes, calibrated_probs)

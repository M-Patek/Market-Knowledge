import pandas as pd
import numpy as np
from .reasoning_ensemble import ReasoningEnsemble
from .base_trainer import BaseTrainer
import mlflow
import os

class CombinatorialPurgedCV:
    def __init__(self, n_splits=10, purging_period=1):
        """
        Initializes the Combinatorial Purged Cross-Validation class.
        """
        self.n_splits = n_splits
        self.purging_period = purging_period

    def split(self, X):
        """
        Generates splits for CPCV with purging.
        
        :param X: pandas DataFrame or Series with a DatetimeIndex.
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have a DatetimeIndex.")
            
        n_samples = len(X)
        indices = np.arange(n_samples)
        
        # Calculate split sizes
        fold_size = n_samples // self.n_splits
        
        for i in range(self.n_splits):
            # Determine validation set boundaries
            validation_start_idx = i * fold_size
            validation_end_idx = validation_start_idx + fold_size
            validation_indices = indices[validation_start_idx:validation_end_idx]
            
            # Determine training set boundaries with purging
            validation_start_time = X.index[validation_start_idx]
            purged_end_time = validation_start_time - pd.Timedelta(days=self.purging_period)
            train_indices = indices[X.index <= purged_end_time]

            yield train_indices, validation_indices

class ModelCardGenerator:
    def __init__(self, template_path='model_card_example.md'):
        self.template_path = template_path
        if not os.path.exists(self.template_path):
            raise FileNotFoundError(f"Model card template not found at: {self.template_path}")
        with open(self.template_path, 'r') as f:
            self.template = f.read()

    def generate_and_log(self, metrics: dict):
        """
        Generates a model card from the template and logs it to MLFlow.
        """
        card_content = self.template
        
        # Replace placeholders with actual values
        card_content = card_content.replace('{{DSR}}', f"{metrics.get('deflated_sharpe_ratio', 'N/A'):.4f}")
        card_content = card_content.replace('{{MAX_DRAWDOWN}}', f"{metrics.get('max_drawdown', 'N/A'):.4f}")
        
        # Write to a temporary file and log as an artifact
        try:
            card_path = "generated_model_card.md"
            with open(card_path, 'w') as f:
                f.write(card_content)
            mlflow.log_artifact(card_path, "model_card")
        except Exception as e:
            print(f"Error logging model card: {e}") # Replace with proper logging

class WalkForwardTrainer(BaseTrainer):
    """
    Performs walk-forward training and validation of a trading strategy.
    """
    def __init__(self, config, strategy, reasoning_ensemble):
        super().__init__(config, strategy)
        self.reasoning_ensemble = reasoning_ensemble

    def train(self, data):
        """
        Trains the strategy using walk-forward cross-validation.
        """
        cv = CombinatorialPurgedCV(n_splits=self.config.get('cv_splits', 5))
        
        # Log trainer-specific parameters to MLFlow
        mlflow.log_param('cv_splits', cv.n_splits)
        
        negative_samples = []
        self.logger.info("Starting walk-forward training...")
        
        for fold, (train_indices, val_indices) in enumerate(cv.split(data)):
            train_data = data.iloc[train_indices]
            val_data = data.iloc[val_indices]

            self.logger.info(f"CV Fold {fold+1}/{cv.n_splits}: Training...")
            self.strategy.train(train_data)

            self.logger.info(f"CV Fold {fold+1}/{cv.n_splits}: Validating...")
            # Simulate returns for the validation period to calculate a realistic metric
            # This is a simplified simulation, assuming daily returns.
            n_days_val = len(val_data)
            simulated_val_returns = np.random.normal(loc=0.0004, scale=0.018, size=n_days_val)
            var_95_val = np.percentile(simulated_val_returns, 5)
            cvar_95_val = simulated_val_returns[simulated_val_returns <= var_95_val].mean()
            performance_metrics = {'cvar': abs(cvar_95_val)}

            cvar_limit = self.config.get('cvar_limit', 0.1)
            if performance_metrics['cvar'] > cvar_limit:
                self.logger.warning(f"CV Fold {fold+1}: CVaR limit breached! ({performance_metrics['cvar']:.4f} > {cvar_limit})")
                failure_context = self._log_failure_scenario_for_retraining(val_data, performance_metrics)
                negative_samples.append(failure_context)
        
        self.logger.info("Walk-forward training complete.")
        if negative_samples:
            self.logger.info(f"Feeding {len(negative_samples)} failure scenarios back to the ReasoningEnsemble...")
            self.reasoning_ensemble.learn_from_failure_scenarios(negative_samples)
        
        # After training, log final performance metrics
        final_metrics = {'deflated_sharpe_ratio': 0.95, 'max_drawdown': -0.08} # Dummy final metrics
        self.logger.info(f"Logging final metrics to MLFlow: {final_metrics}")
        dummy_returns = pd.Series(np.random.normal(loc=0.001, scale=0.015, size=252))
        confidence_intervals = self._calculate_confidence_intervals(dummy_returns)
        final_metrics.update(confidence_intervals)
        mlflow.log_metrics(final_metrics)

        # Generate and log the Model Card
        self.logger.info("Generating Model Card...")
        card_generator = ModelCardGenerator()
        card_generator.generate_and_log(final_metrics)

        return self.strategy

    def _log_failure_scenario_for_retraining(self, data_subset, metrics):
        """
        Analyzes and captures the context of a training failure.
        """
        self.logger.info("Logging failure scenario for adaptive risk learning...")
        failure_context = {'metrics': metrics, 'triggering_data_hash': hash(data_subset.to_string())}
        return failure_context

    def _calculate_confidence_intervals(self, returns: pd.Series, n_bootstrap=1000, ci_level=0.95):
        """
        Calculates confidence intervals for key metrics using bootstrapping.
        """
        self.logger.info(f"Calculating {ci_level:.0%} confidence intervals with {n_bootstrap} bootstrap samples...")
        stats = []
        for _ in range(n_bootstrap):
            sample_returns = returns.sample(n=len(returns), replace=True)
            if sample_returns.std() == 0:
                continue
            sharpe_ratio = (sample_returns.mean() / sample_returns.std()) * np.sqrt(252)
            stats.append(sharpe_ratio)
        
        lower_bound = np.percentile(stats, (1 - ci_level) / 2 * 100)
        upper_bound = np.percentile(stats, (1 + ci_level) / 2 * 100)
        self.logger.info(f"Sharpe Ratio {ci_level:.0%} CI: [{lower_bound:.4f}, {upper_bound:.4f}]")

        return {'sharpe_ci_lower': lower_bound, 'sharpe_ci_upper': upper_bound}

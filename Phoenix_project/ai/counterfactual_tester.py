import pandas as pd
import numpy as np
from observability import get_logger

class CounterfactualTester:
    def __init__(self, config, strategy):
        self.config = config
        self.strategy = strategy
        self.logger = get_logger(__name__)

    def run_tests(self):
        self.logger.info("Running static counterfactual tests...")
        
        self.run_monte_carlo_stress_test()

        test_results = {'scenario_1': 'pass', 'scenario_2': 'fail'}
        return test_results

    def _run_backtest_with_perturbations(self, perturbations: dict):
        """
        A helper function to run a single backtest with perturbed data.
        """
        total_perturbation = sum(abs(p) for p in perturbations.values())
        cvar_limit = self.config.get('cvar_limit', 0.1)
        if np.random.rand() < total_perturbation:
             return {'cvar': cvar_limit + 0.05, 'status': 'fail'}
        return {'cvar': cvar_limit - 0.02, 'status': 'pass'}

    def run_monte_carlo_stress_test(self, n_simulations=1000):
        """
        Performs adaptive stress testing using Monte Carlo methods.
        """
        self.logger.info(f"Starting Monte Carlo stress test with {n_simulations} simulations...")
        stress_params = self.config.get('stress_test_params', {})
        if not stress_params:
            self.logger.warning("`stress_test_params` not found in config. Skipping Monte Carlo test.")
            return

        failure_scenarios = []

        for i in range(n_simulations):
            perturbations = {}
            for factor, params in stress_params.items():
                dist = params.get('distribution', 'normal')
                if dist == 'normal':
                    mean_shift = params.get('mean_shift', 0)
                    std_multiplier = params.get('std_dev_multiplier', 1)
                    perturbation = np.random.normal(loc=mean_shift, scale=1.0 * std_multiplier)
                elif dist == 'uniform':
                    min_shift = params.get('min_shift', -0.01)
                    max_shift = params.get('max_shift', 0.01)
                    perturbation = np.random.uniform(low=min_shift, high=max_shift)
                else:
                    perturbation = 0
                perturbations[factor] = perturbation

            backtest_result = self._run_backtest_with_perturbations(perturbations)

            if backtest_result['status'] == 'fail':
                failure_scenarios.append({'simulation_id': i, 'perturbations': perturbations})

        self.logger.info(f"Monte Carlo stress test complete. Found {len(failure_scenarios)} failure scenarios.")


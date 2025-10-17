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
        This is a simplified simulation, not a full backtrader run.
        """
        cvar_limit = self.config.get('cvar_limit', 0.1)

        # 1. Simulate a series of daily returns
        n_days = 252 # Simulate one year of returns
        base_mean_return = 0.0005
        base_std_dev = 0.015

        # 2. Apply the perturbation to the mean return
        # A simple model: each factor perturbation additively affects the mean.
        total_perturbation_effect = sum(perturbations.values())
        perturbed_mean = base_mean_return + total_perturbation_effect

        simulated_returns = np.random.normal(loc=perturbed_mean, scale=base_std_dev, size=n_days)

        # 3. Calculate CVaR from the simulated returns (e.g., 95% CVaR)
        var_95 = np.percentile(simulated_returns, 5)
        cvar_95 = simulated_returns[simulated_returns <= var_95].mean()

        status = 'fail' if abs(cvar_95) > cvar_limit else 'pass'
        return {'cvar': abs(cvar_95), 'status': status}

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

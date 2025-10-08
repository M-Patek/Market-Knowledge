# optimizer.py

import logging
from typing import Dict, Any
import datetime
from datetime import timedelta
import asyncio
import optuna
import backtrader as bt
import pandas as pd
from phoenix_project import StrategyConfig, RomanLegionStrategy, generate_html_report, precompute_sentiments, precompute_asset_analyses
from ai.client import AIClient


optuna.logging.set_verbosity(optuna.logging.WARNING)

class Optimizer:
    """
    Manages the Optuna-based walk-forward optimization process.
    """

    def __init__(self, config: StrategyConfig, all_aligned_data: Dict[str, Any], ai_client: AIClient | None):
        """
        Initializes the Optimizer.

        Args:
            config: The main strategy configuration object.
            all_aligned_data: The complete, aligned dataset for the backtest period.
            ai_client: The AI client for report generation and pre-computation.
        """
        self.logger = logging.getLogger("PhoenixProject.Optimizer")
        self.config = config
        self.all_aligned_data = all_aligned_data
        self.ai_client = ai_client
        self.study_name = self.config.optimizer.study_name
        self.storage_url = f"sqlite:///{self.study_name}.db"

    def _objective(self, trial: optuna.trial.Trial, start_date: datetime.date, end_date: datetime.date) -> float:
        """The function for Optuna to optimize, which runs a single backtest."""
        try:
            # 1. Suggest hyperparameters from the config file
            params_to_tune = {}
            for param_name, details in self.config.optimizer.parameters.items():
                if details['type'] == 'int':
                    params_to_tune[param_name] = trial.suggest_int(
                        param_name, details['low'], details['high'], step=details['step']
                    )
                elif details['type'] == 'float':
                     params_to_tune[param_name] = trial.suggest_float(
                        param_name, details['low'], details['high'], step=details['step']
                    )

            # 2. Create a temporary config for this trial
            temp_config = self.config.copy(deep=True)
            for param_name, value in params_to_tune.items():
                setattr(temp_config, param_name, value)

            # 3. Set up and run the isolated backtest
            cerebro = bt.Cerebro(stdstats=False)
            cerebro.broker.setcash(temp_config.initial_cash)

            for ticker in temp_config.asset_universe:
                df = self.all_aligned_data["asset_universe_df"].xs(ticker, level=1, axis=1).copy()
                df.columns = [col.lower() for col in df.columns]
                data_feed = bt.feeds.PandasData(dataname=df, fromdate=start_date, todate=end_date, name=ticker)
                cerebro.adddata(data_feed)

            cerebro.addstrategy(
                RomanLegionStrategy, config=temp_config,
                vix_data=self.all_aligned_data["vix"],
                treasury_yield_data=self.all_aligned_data["treasury_yield"],
                market_breadth_data=self.all_aligned_data["market_breadth"],
                sentiment_data={}, asset_analysis_data={} # AI data is off for optimization phase
            )
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio', annualize=True)
            
            results = cerebro.run()
            sharpe_ratio = results[0].analyzers.sharpe_ratio.get_analysis().get('sharperatio', 0.0)
            return sharpe_ratio if sharpe_ratio is not None else 0.0

        except Exception as e:
            self.logger.error(f"Trial {trial.number} failed with error: {e}")
            return -1e9 # Return a very bad score to penalize failing trials

    def run_optimization(self):
        """The main entry point to start the walk-forward optimization process."""
        self.logger.info(f"--- Launching Optuna Walk-Forward Optimization (Study: '{self.study_name}') ---")
        
        wf_config = self.config.walk_forward
        train_delta = timedelta(days=wf_config['train_days'])
        test_delta = timedelta(days=wf_config['test_days'])
        step_delta = timedelta(days=wf_config['step_days'])
        
        full_start_date = self.config.start_date
        full_end_date = self.config.end_date

        out_of_sample_results = []
        current_start = full_start_date
        window_num = 0

        while current_start + train_delta + test_delta <= full_end_date:
            window_num += 1
            train_start = current_start
            train_end = train_start + train_delta
            test_start = train_end + timedelta(days=1)
            test_end = test_start + test_delta - timedelta(days=1)
            
            self.logger.info(f"--- W-F Window {window_num}: Train=[{train_start} to {train_end}] ---")

            # 2. Create and run the Optuna study for the training period
            study = optuna.create_study(
                study_name=f"{self.study_name}_window_{window_num}",
                storage=self.storage_url,
                direction="maximize",
                load_if_exists=True # Allows resuming
            )
            
            study.optimize(
                lambda trial: self._objective(trial, train_start, train_end),
                n_trials=self.config.optimizer.n_trials
            )
            
            best_params = study.best_params
            self.logger.info(f"Window {window_num} Best Sharpe: {study.best_value:.3f}, Best Params: {best_params}")

            # 3. Run validation backtest on the test period with the best params
            self.logger.info(f"--- W-F Window {window_num}: Test=[{test_start} to {test_end}] ---")
            val_cerebro = bt.Cerebro()
            run_audit_files = []
            
            temp_config = self.config.copy(deep=True)
            for param, value in best_params.items():
                setattr(temp_config, param, value)

            master_dates = pd.date_range(start=test_start, end=test_end).date
            sentiment_lookup, sentiment_audits = asyncio.run(precompute_sentiments(self.ai_client, master_dates)) if self.ai_client else ({}, [])
            asset_analysis_lookup, asset_audits = asyncio.run(precompute_asset_analyses(self.ai_client, master_dates, self.config.asset_universe)) if self.ai_client else ({}, [])
            
            if sentiment_audits: run_audit_files.extend(sentiment_audits)
            if asset_audits: run_audit_files.extend(asset_audits)
            
            val_cerebro.addstrategy(
                RomanLegionStrategy, config=temp_config,
                vix_data=self.all_aligned_data["vix"],
                treasury_yield_data=self.all_aligned_data["treasury_yield"],
                market_breadth_data=self.all_aligned_data["market_breadth"],
                sentiment_data=sentiment_lookup,
                asset_analysis_data=asset_analysis_lookup
            )

            val_cerebro.broker.setcash(self.config.initial_cash)
            for ticker in self.config.asset_universe:
                df = self.all_aligned_data["asset_universe_df"].xs(ticker, level=1, axis=1).copy()
                df.columns = [col.lower() for col in df.columns]
                data_feed = bt.feeds.PandasData(dataname=df, fromdate=test_start, todate=test_end, name=ticker)
                val_cerebro.adddata(data_feed)

            val_cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio', annualize=True)
            val_cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            
            results = val_cerebro.run()
            strat = results[0]
            out_of_sample_results.append(strat)
            asyncio.run(generate_html_report(val_cerebro, strat, self.ai_client, run_audit_files, report_filename=f"phoenix_report_wf_{window_num}.html"))

            current_start += step_delta

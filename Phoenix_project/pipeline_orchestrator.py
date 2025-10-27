import yaml
from observability import get_logger
import mlflow
import subprocess
import redis
from typing import Dict, Any

class PipelineOrchestrator:
    def __init__(self, config_path):
        self.logger = get_logger(__name__)
        self.config_path = config_path
        self.config = self.load_config()
        # [Epic 2.2] Connect to Redis for persistent queuing
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

    def load_config(self):
        """Loads the pipeline configuration from a YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found at {self.config_path}")
            return {}
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            return {}

    def run(self):
        """
        Runs the full ML pipeline.
        V2.0: Now includes a qualification gate for model promotion.
        """
        mlflow.set_experiment(self.config.get('experiment_name', 'DefaultExperiment'))
        
        with mlflow.start_run() as run:
            self.logger.info(f"Starting pipeline run with MLflow run_id: {run.info.run_id}")
            
            # Log configuration
            mlflow.log_artifact(self.config_path, "config")

            # TODO: Add calls to other components like data loading, training, etc.
            # For now, we'll simulate the output of a training run.
            challenger_metrics = {
                "sharpe_ratio_dsr": 0.95,
                "max_drawdown": -0.08
            }
            mlflow.log_metrics(challenger_metrics)
            challenger_model_id = run.info.run_id

            # --- [Sub-Task 1.1.1: Qualification Gate] ---
            # In a real system, we would fetch the champion model's metrics from a model registry.
            champion_metrics = {"sharpe_ratio_dsr": 0.90, "max_drawdown": -0.10} # Mocked metrics

            if self._is_challenger_qualified(champion_metrics, challenger_metrics):
                self.logger.info(f"Challenger model '{challenger_model_id}' PASSED qualification gate.")
                self.redis_client.lpush('AWAITING_CANARY', challenger_model_id)
                self.logger.info(f"Model '{challenger_model_id}' pushed to Redis queue 'AWAITING_CANARY'.")
            else:
                self.logger.warning(f"Challenger model '{challenger_model_id}' FAILED qualification gate. Not promoting.")

            self.logger.info("Pipeline orchestration complete.")

    def _is_challenger_qualified(self, champion_metrics: Dict[str, float], challenger_metrics: Dict[str, float]) -> bool:
        """Compares challenger metrics against the champion to see if it qualifies for promotion."""
        challenger_sharpe = challenger_metrics.get('sharpe_ratio_dsr', 0.0)
        champion_sharpe = champion_metrics.get('sharpe_ratio_dsr', 0.0)
        challenger_drawdown = challenger_metrics.get('max_drawdown', -1.0)
        champion_drawdown = champion_metrics.get('max_drawdown', -1.0)

        # Qualification criteria: Higher DSR Sharpe AND lower (less negative) drawdown.
        return challenger_sharpe > champion_sharpe and challenger_drawdown > champion_drawdown

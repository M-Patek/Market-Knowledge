import yaml
from observability import get_logger
import mlflow
import subprocess

class PipelineOrchestrator:
    def __init__(self, config_path):
        self.logger = get_logger(__name__)
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def run(self):
        """
        Runs the full ML pipeline.
        """
        mlflow.set_experiment(self.config.get('experiment_name', 'DefaultExperiment'))
        
        with mlflow.start_run() as run:
            self.logger.info(f"Starting pipeline orchestration with MLFlow Run ID: {run.info.run_id}")
            
            # Log Git Commit Hash
            try:
                commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
                mlflow.log_param('git_commit_hash', commit_hash)
            except Exception as e:
                self.logger.error(f"Could not get git commit hash: {e}")

            # Log config file as an artifact
            mlflow.log_artifact(self.config_path, "config")

            # TODO: Add calls to other components like data loading, training, etc.
            self.logger.info("Pipeline orchestration complete.")


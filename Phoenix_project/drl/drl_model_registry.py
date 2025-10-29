# drl/drl_model_registry.py
import mlflow
import mlflow.pytorch
import torch
from typing import Dict, Any, List
import os

class DRLModelRegistry:
    """
    Handles the serialization and version control of the complex
    multi-agent DRL models using MLflow.
    """

    def __init__(self, tracking_uri: str = None):
        """
        Initializes the registry and connects to the MLflow tracking server.

        Args:
            tracking_uri (str, optional): The MLflow tracking URI. If None,
                                          MLflow will use its default.
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        self.tracking_uri = tracking_uri
        print(f"DRLModelRegistry initialized. Tracking URI: {mlflow.get_tracking_uri()}")

    def save_models(self, 
                    run_name: str, 
                    config: Dict[str, Any], 
                    models: Dict[str, torch.nn.Module]) -> str:
        """
        Starts a new MLflow run, logs parameters, and saves all model
        state dictionaries as artifacts.

        Args:
            run_name (str): The name for the MLflow run.
            config (Dict[str, Any]): The trainer's configuration dictionary.
            models (Dict[str, torch.nn.Module]): A dictionary mapping
                a model's name (e.g., 'strategic_critic') to the
                PyTorch model object.

        Returns:
            str: The run_id of the completed MLflow run.
        """
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            print(f"Starting MLflow run: {run_name} (ID: {run_id})")
            
            # Log all configuration parameters
            # MLflow can have issues logging complex nested dicts, flatten first if needed
            flat_config = pd.json_normalize(config, sep='_').to_dict(orient='records')[0]
            mlflow.log_params(flat_config)

            # Save each model using the mlflow.pytorch flavor
            for name, model in models.items():
                if model is not None:
                    print(f"Logging model: {name}")
                    # This saves the model in a format MLflow understands
                    try:
                        mlflow.pytorch.log_model(model, artifact_path=name)
                    except Exception as e:
                        print(f"Warning: Could not log model {name} with mlflow.pytorch.log_model. Error: {e}")
                        print("Attempting to save state_dict as artifact.")
                        with torch.no_grad():
                            temp_path = f"{name}.pth"
                            torch.save(model.state_dict(), temp_path)
                            mlflow.log_artifact(temp_path, artifact_path=f"{name}_statedict")
                            os.remove(temp_path)

            
            print(f"Successfully saved all models for run {run_id}.")
            return run_id

    def load_models(self, 
                    run_id: str, 
                    model_names: List[str]) -> Dict[str, torch.nn.Module]:
        """
        Loads a set of models from a specific MLflow run_id.

        Args:
            run_id (str): The MLflow run_id to load from.
            model_names (List[str]): A list of the model artifact names
                                     (e.g., ['strategic_critic', 'alpha_actor']).

        Returns:
            Dict[str, torch.nn.Module]: A dictionary mapping the model's
                                        name to the loaded PyTorch model.
        """
        print(f"Loading models from MLflow run_id: {run_id}")
        loaded_models = {}
        
        for name in model_names:
            try:
                # Construct the URI for the model artifact
                model_uri = f"runs:/{run_id}/{name}"
                print(f"Loading model: {name} from {model_uri}")
                
                # Load the model using the mlflow.pytorch flavor
                loaded_models[name] = mlflow.pytorch.load_model(model_uri)
            except Exception as e:
                print(f"Failed to load model '{name}' from run '{run_id}'. Error: {e}")
                # Add fallback for state_dict if mlflow.pytorch fails
                try:
                    print(f"Attempting to load state_dict for {name}...")
                    client = mlflow.tracking.MlflowClient()
                    local_path = client.download_artifacts(run_id, f"{name}_statedict/{name}.pth")
                    
                    # This part is tricky, as we need to know the model class to instantiate.
                    # This is a limitation of not using mlflow.pytorch.log_model
                    print(f"Error: Cannot automatically load {name} from state_dict without model class definition.")
                    print("Please ensure models were saved with mlflow.pytorch.log_model for loading.")
                    raise
                except:
                    raise e
        
        print("Successfully loaded all requested models.")
        return loaded_models

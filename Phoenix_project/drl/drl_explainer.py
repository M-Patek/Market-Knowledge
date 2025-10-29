# drl/drl_explainer.py
import shap
import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from .drl_model_registry import DRLModelRegistry

class DRLExplainer:
    """
    (Task 3.2) Implements explainability for DRL agents using SHAP.
    This class can be used to analyze the feature importance for an
    agent's decisions at critical points.
    """

    def __init__(self, model_registry: DRLModelRegistry, feature_names: List[str]):
        """
        Initializes the explainer.

        Args:
            model_registry (DRLModelRegistry): The registry to load models from.
            feature_names (List[str]): The ordered list of feature names corresponding
                                       to the observation space columns.
        """
        self.model_registry = model_registry
        self.feature_names = feature_names
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def explain_actor_decision(self,
                               run_id: str,
                               actor_name: str,
                               background_states: np.ndarray,
                               decision_states: np.ndarray) -> Dict[str, Any]:
        """
        Calculates SHAP values for a specific actor's decisions.

        Args:
            run_id (str): The MLflow run_id where the model is stored.
            actor_name (str): The name of the actor model artifact.
            background_states (np.ndarray): A sample of typical states to serve
                                            as the background distribution for SHAP.
            decision_states (np.ndarray): The specific state(s) where we want
                                          to explain the agent's decision.

        Returns:
            Dict[str, Any]: A dictionary containing the SHAP values and base values.
        """
        print(f"--- Starting DRL Explanation for {actor_name} from run {run_id} ---")
        try:
            # 1. Load the actor network
            actor_network = self.model_registry.load_models(run_id, [actor_name])[actor_name]
            actor_network.to(self.device)
            actor_network.eval()

            # 2. Convert numpy arrays to torch tensors
            if background_states.ndim == 1:
                background_states = background_states.reshape(1, -1)
            if decision_states.ndim == 1:
                decision_states = decision_states.reshape(1, -1)
                
            background_tensor = torch.tensor(background_states, dtype=torch.float32).to(self.device)
            decision_tensor = torch.tensor(decision_states, dtype=torch.float32).to(self.device)

            # 3. Instantiate the SHAP DeepExplainer
            # It takes the model and a background data distribution
            explainer = shap.DeepExplainer(actor_network, background_tensor)

            # 4. Calculate SHAP values for the critical decision states
            print("Calculating SHAP values...")
            shap_values = explainer.shap_values(decision_tensor)
            
            print("Explanation complete.")
            
            # For visualization and interpretation
            return {
                "shap_values": shap_values,
                "base_values": explainer.expected_value,
                "decision_states": decision_states,
                "feature_names": self.feature_names
            }

        except Exception as e:
            print(f"Failed to generate DRL explanation: {e}")
            raise

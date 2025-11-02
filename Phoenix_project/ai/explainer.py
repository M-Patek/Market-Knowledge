import shap
import torch
from typing import Dict, Any, List

from ..monitor.logging import get_logger
from .agents.base_agent import BaseAgent

logger = get_logger(__name__)

class DRLExplainer:
    """
    Uses SHAP (SHapley Additive exPlanations) to explain the
    decisions of a trained DRL agent's network.
    
    It helps answer: "Why did the agent take this action?"
    """

    def __init__(self, agent: BaseAgent, background_data: torch.Tensor, feature_names: List[str]):
        """
        Initializes the explainer.
        
        Args:
            agent (BaseAgent): The trained DRL agent (with its network).
            background_data (torch.Tensor): A sample of 'typical' observations
                                            used as the baseline for SHAP.
            feature_names (List[str]): Names of the features in the observation space.
        """
        self.agent = agent
        self.network = agent.network
        self.network.eval() # Set network to evaluation mode
        
        self.background_data = background_data
        self.feature_names = feature_names
        
        logger.info(f"DRLExplainer initialized for agent: {agent.agent_id}")

        # SHAP requires a function that takes a (N, D) numpy array
        # and returns a (N, K) numpy array of outputs (where K=1 for us).
        def shap_predict_function(observations_np):
            """Wrapper for SHAP DeepExplainer."""
            try:
                # 1. Convert numpy array to torch tensor
                observations_tensor = torch.tensor(observations_np, dtype=torch.float32)
                
                # 2. Run tensor through the network
                # We want to explain the *mean* of the action distribution,
                # not a random sample.
                with torch.no_grad():
                    action_distribution = self.network(observations_tensor)
                    # Get the mean of the distribution (the most likely action)
                    mean_action = action_distribution.mean
                
                # 3. Return as numpy array
                return mean_action.cpu().numpy()
                
            except Exception as e:
                logger.error(f"SHAP prediction function failed: {e}", exc_info=True)
                # Return an array of zeros with the expected shape
                return np.zeros((observations_np.shape[0], self.agent.action_space.shape[0]))

        # Initialize the SHAP DeepExplainer
        try:
            self.explainer = shap.DeepExplainer(
                self.network, # The model (PyTorch network)
                self.background_data # The background data (PyTorch tensor)
            )
            logger.info("SHAP DeepExplainer successfully initialized.")
        except Exception as e:
            logger.warning(f"Failed to initialize DeepExplainer. Falling back to KernelExplainer. Error: {e}")
            # Fallback to KernelExplainer (slower, but model-agnostic)
            self.explainer = shap.KernelExplainer(
                shap_predict_function, # The function (numpy in, numpy out)
                self.background_data.cpu().numpy() # Background data (numpy)
            )
            logger.info("SHAP KernelExplainer successfully initialized as fallback.")

    def explain_decision(self, observation: torch.Tensor) -> Dict[str, float]:
        """
        Generates SHAP values for a single observation.
        
        Args:
            observation (torch.Tensor): The specific state observation
                                        to explain (1, D) tensor.
                                        
        Returns:
            Dict[str, float]: A dictionary mapping feature_name -> shap_value.
        """
        if observation.dim() == 1:
            observation = observation.unsqueeze(0) # Add batch dimension
            
        logger.debug(f"Generating SHAP explanation for observation: {observation.shape}")

        try:
            # Calculate SHAP values
            # shap_values is a list (one per output), but our output is 1D (action)
            # So shap_values[0] will be (1, D) numpy array
            shap_values = self.explainer.shap_values(observation)
            
            if isinstance(self.explainer, shap.KernelExplainer):
                # KernelExplainer output is just the array
                shap_values_flat = shap_values[0]
            else:
                # DeepExplainer output is a list
                shap_values_flat = shap_values[0][0]
                
            if len(self.feature_names) != len(shap_values_flat):
                logger.error(f"Mismatch in feature names ({len(self.feature_names)}) and "
                               f"SHAP values ({len(shap_values_flat)}).")
                return {"error": "Feature name/SHAP value count mismatch."}

            # Map feature names to their SHAP values
            explanation = dict(zip(self.feature_names, shap_values_flat))
            
            # Sort by absolute SHAP value for importance
            sorted_explanation = {k: v for k, v in sorted(
                explanation.items(), 
                key=lambda item: abs(item[1]), 
                reverse=True
            )}
            
            return sorted_explanation

        except Exception as e:
            logger.error(f"Failed to generate SHAP explanation: {e}", exc_info=True)
            return {"error": str(e)}

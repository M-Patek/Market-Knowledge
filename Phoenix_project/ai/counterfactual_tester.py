# (Assuming existing imports like 'logging')
import logging
import torch
import numpy as np
# 修复：[FIX-1] 路径从 'drl.drl_model_registry' 更改为 '..models.registry'
# 并将类名从 'DRLModelRegistry' 更改为 'ModelRegistry'
from ..models.registry import ModelRegistry
from typing import Dict, Any

class CounterfactualTester:
    def __init__(self, config: Dict[str, Any]):
        # (Assuming existing __init__ logic)
        self.logger = logging.getLogger("PhoenixProject.CounterfactualTester")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run_ensemble_test(self, model, test_data, scenarios):
        # (Assuming existing methods for the ReasoningEnsemble)
        self.logger.info("Running counterfactuals for ReasoningEnsemble...")
        # ... existing logic ...
        return {}

    def run_drl_adversarial_test(self, 
                                 # 修复：[FIX-1]
                                 model_registry: ModelRegistry, 
                                 run_id: str, 
                                 actor_name: str, 
                                 test_states: np.ndarray, 
                                 epsilon: float = 0.01) -> Dict[str, float]:
        """
        (Task 3.2) Performs an FGSM adversarial attack on a DRL policy network.
        
        Measures how much the agent's action changes in response to a
        tiny, adversarially-chosen perturbation in the state.
        """
        self.logger.info(f"Running DRL adversarial test on {actor_name} from run {run_id}...")
        
        try:
            # 1. Load the target policy network
            policy_network = model_registry.load_models(run_id, [actor_name])[actor_name]
            policy_network.to(self.device)
            policy_network.eval()
            
            # 2. Prepare states tensor and enable gradients
            if test_states.ndim == 1:
                test_states = test_states.reshape(1, -1) # Ensure 2D
                
            states_tensor = torch.tensor(test_states, dtype=torch.float32, device=self.device)
            states_tensor.requires_grad = True
            
            # 3. Get baseline (original) actions
            baseline_actions = policy_network(states_tensor)
            
            # 4. Generate perturbation (FGSM)
            # We need a "loss" to backprop from. We'll use the mean of the action
            # (or sum) as a simple scalar value to get the gradient.
            loss = baseline_actions.mean()
            policy_network.zero_grad()
            loss.backward()
            
            if states_tensor.grad is None:
                self.logger.error("Could not compute gradient for FGSM. Ensure model has trainable parameters.")
                return {"error": "Gradient is None."}

            # Get the sign of the gradient
            grad_sign = states_tensor.grad.data.sign()
            perturbed_states = states_tensor + epsilon * grad_sign
            
            # 5. Get actions from perturbed states
            with torch.no_grad():
                perturbed_actions = policy_network(perturbed_states)
            
            # 6. Calculate and report the magnitude of the action change
            action_diff = torch.norm(baseline_actions - perturbed_actions, p=2, dim=1)
            avg_perturbation = action_diff.mean().item()
            
            self.logger.info(f"Adversarial test complete. Epsilon: {epsilon}, Avg. Action Perturbation: {avg_perturbation:.4f}")
            
            return {
                "epsilon": epsilon,
                "avg_action_perturbation": avg_perturbation,
                "max_action_perturbation": action_diff.max().item()
            }

        except Exception as e:
            self.logger.error(f"Failed DRL adversarial test: {e}")
            return {"error": str(e)}

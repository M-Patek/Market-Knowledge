import yaml
import logging
from typing import List, Dict, Any
# Assuming PipelineState is accessible for type hinting
# from pipeline_state import PipelineState 

logger = logging.getLogger(__name__)

class L3_Rules_Engine:
    """
    (L4 Task 3) Loads and applies rules from a YAML file based on L3 meta-cognition output.
    """
    def __init__(self, rules_yaml_path: str):
        self.rules = self._load_rules(rules_yaml_path)
        logger.info(f"L3_Rules_Engine initialized with {len(self.rules)} rules from {rules_yaml_path}.")

    def _load_rules(self, file_path: str) -> List[Dict[str, Any]]:
        """Loads and validates the rules from the specified YAML file."""
        try:
            with open(file_path, 'r') as f:
                rules = yaml.safe_load(f)
            # Basic validation
            if not isinstance(rules, list):
                raise ValueError("Rules YAML must contain a list of rule objects.")
            return rules
        except Exception as e:
            logger.error(f"Failed to load or parse L3 rules from {file_path}: {e}", exc_info=True)
            return []

    def apply_rules(self, meta_log: Dict[str, Any], pipeline_state: Any) -> List[Dict[str, Any]]:
        """
        Evaluates the loaded rules against the current meta_log and pipeline_state.

        Args:
            meta_log: The output from the MetacognitiveAgent's meta_update method.
            pipeline_state: The current PipelineState object.

        Returns:
            A list of action dictionaries for all triggered rules.
        """
        triggered_actions = []
        
        # Combine data sources for evaluation context
        eval_context = {**meta_log, **pipeline_state.__dict__}

        for rule in self.rules:
            try:
                condition = rule.get("condition")
                action = rule.get("action")
                if not all([condition, action]):
                    logger.warning(f"Skipping malformed rule: {rule}")
                    continue

                # WARNING: eval() is used for simplicity. A real production system
                # should use a safer expression evaluation library (e.g., py-expression-eval).
                if eval(condition, {}, eval_context):
                    logger.info(f"L3 Rule '{rule.get('name')}' triggered by condition: '{condition}'")
                    triggered_actions.append(action)
            
            except Exception as e:
                logger.error(f"Error evaluating L3 rule '{rule.get('name')}': {e}")

        return triggered_actions

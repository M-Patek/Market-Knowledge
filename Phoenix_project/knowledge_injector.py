# knowledge_injector.py
# This module is responsible for L3 meta-cognition.
# It receives rules from the MetaCognitiveAgent and injects them
# into the L1/L2 systems.

import os
import json
import logging
from typing import List, Dict, Any
from ai.prompt_manager import PromptManager
from ai.tabular_db_client import TabularDBClient # Placeholder

# Import the store we built in Phase 2 (Task 2.2)
try:
    from ai.bayesian_fusion_engine import SourceCredibilityStore
    # We'd also need to initialize it, which is complex without DI.
    # For this task, we'll assume a way to access it.
except ImportError:
    logger.warning("Could not import SourceCredibilityStore in KnowledgeInjector.")
    SourceCredibilityStore = None 

# Configure logging
logger = logging.getLogger("PhoenixProject.KnowledgeInjector")


def inject_rules(rules: List[Dict[str, Any]]):
    """
    Task 4.3: Main function to parse and inject L3 rules.
    
    A "rule" is a dictionary, e.g.:
    {
      "name": "Down-weight technicals in high-vol",
      "action_type": "adjust_credibility",
      "target_agent": "technical_analyst",
      "adjustment_factor": -0.2,
      "conditions": ["market_volatility > 0.8"]
    }
    or
    {
      "name": "Add constraint for analyst bias",
      "action_type": "update_prompt",
      "target_agent": "fundamental_analyst",
      "description": "Always double-check revenue projections against 3rd-party data."
    }
    """
    logger.info(f"KnowledgeInjector processing {len(rules)} new rules...")
    
    # Initialize the store to handle credibility adjustments
    credibility_store = SourceCredibilityStore(TabularDBClient()) # Placeholder init
    # --- Task 3.3: Initialize PromptManager ---
    db_client_placeholder = TabularDBClient() # In a real scenario, this would be properly configured and injected.
    prompt_manager = PromptManager(db_client=db_client_placeholder)
    
    for rule in rules:
        try:
            action_type = rule.get('action_type')
            target_agent = rule.get('target_agent')
            
            if action_type == "adjust_credibility" and target_agent:
                # --- Injection Path 2 (Update Credibility) ---
                # This is a simplified example. A real one would be more complex.
                factor = float(rule.get('adjustment_factor', 0.0))
                if factor > 0:
                    # Simulate adding 'success' data
                    credibility_store.update_credibility(target_agent, success=True)
                else:
                    # Simulate adding 'failure' data
                    credibility_store.update_credibility(target_agent, success=False)
                logger.info(f"Injected credibility adjustment for: {target_agent}")
            
            elif action_type == "update_prompt" and target_agent:
                # --- Injection Path 1 (Update Prompt) ---
                rule_text = rule.get('description', 'No description provided.')
                # --- Task 3.3: Refactor to use PromptManager ---
                try:
                    # 1. Get the prompt from the external store
                    config = prompt_manager.get_prompt(target_agent)
                    if 'permanent_constraints' not in config:
                        config['permanent_constraints'] = []
                    
                    if rule_text not in config['permanent_constraints']:
                        config['permanent_constraints'].append(rule_text)
                        # 2. Update the prompt in the external store
                        prompt_manager.update_prompt(target_agent, config)
                        logger.info(f"Injected new constraint into: {target_agent} via PromptManager.")
                    else:
                        logger.info(f"Constraint already exists in {target_agent}. Skipping update.")
                except Exception as e:
                    logger.warning(f"Could not find/update prompt for injection: {target_agent}. Error: {e}")

        except Exception as e:
            logger.error(f"Failed to inject rule {rule.get('name')}: {e}")

# Example of how this might be run
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Running KnowledgeInjector standalone test...")
    
    mock_rules = [
        {
          "name": "Add constraint for analyst bias",
          "action_type": "update_prompt",
          "target_agent": "fundamental_analyst",
          "description": "Always double-check revenue projections against 3rd-party data."
        },
        {
          "name": "Down-weight technicals in high-vol",
          "action_type": "adjust_credibility",
          "target_agent": "technical_analyst",
          "adjustment_factor": -0.2,
          "conditions": ["market_volatility > 0.8"]
        }
    ]
    
    inject_rules(mock_rules)

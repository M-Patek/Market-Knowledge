import asyncio
import logging
from typing import List, Dict, Any
import yaml
import json

# Internal dependencies
from audit_manager import AuditManager
from api.gemini_pool_manager import query_model # Task 0.3

logger = logging.getLogger("PhoenixProject.MetaCognitiveAgent")


class MetaCognitiveAgent:
    """
    The L3 agent responsible for Causal Inference and Heuristic Rule Generation.
    (Task 2.2 in original doc, Task 4.2 in new spec)
    """
    def __init__(self, audit_manager: AuditManager):
        self.audit_manager = audit_manager
        logger.info("L3 MetaCognitiveAgent initialized.")

    async def run_analysis(self, lookback_days: int) -> List[Dict[str, Any]]:
        """
        Core logic of the L3 agent (Task 4.2).
        """
        logger.info(f"L3 MetaCognitiveAgent: Starting analysis for last {lookback_days} days.")
        
        # 1. Fetch logs with P&L
        logs = self.audit_manager.fetch_logs_with_pnl(lookback_days)
        
        if not logs:
            logger.info("L3 MetaCognitiveAgent: No logs with P&L found. Skipping analysis.")
            return []
        
        # 2. Build Causal Inference Prompt
        prompt = self._build_causal_inference_prompt(logs)
        
        # 3. Call LLM
        raw_yaml_output = await query_model(prompt, agent_name="metacognitive_agent")
        logger.debug(f"L3 MetaCognitiveAgent raw output: {raw_yaml_output}")
        
        # 4. Parse output and return rules
        rules = self._parse_yaml_rules(raw_yaml_output)
        
        logger.info(f"L3 MetaCognitiveAgent: Analysis complete. Generated {len(rules)} new rules.")
        return rules

    def _build_causal_inference_prompt(self, logs: List[Dict[str, Any]]) -> str:
        logger.info(f"Building causal inference prompt for {len(logs)} log entries...")
        
        # Convert logs to a more compact JSON string for the prompt
        log_summary = json.dumps(logs, default=str)

        # Task 4.2: Specialized Causal Inference Prompt
        prompt = f"""
You are a Meta-Cognitive Causal Inference Agent. Your task is to analyze a list of historical trading decisions to find recurrent logical patterns that lead to significant profit or loss.

**Input Data:**
A JSON list of decision logs. Each log contains:
- 'decision_id': A unique identifier.
- 'l1_evidence_items': A list of evidence objects, EACH with a 'reasoning_chain', 'source', and 'score'.
- 'l2_fusion_result': The final L2 probability.
- 'pnl_result': The final profit or loss (the ground truth).

**Your Mission:**
1.  Focus on the 'l1_evidence_items' and their 'reasoning_chain' fields.
2.  Correlate the reasoning, sources, and scores with the final 'pnl_result'.
3.  Identify 1-3 high-conviction, recurrent patterns of error or success.

**Mandatory Output Format (YAML):**
You MUST summarize your findings as 1-3 Heuristic Rules in the following YAML format. Do not output ANY text, explanation, or preamble outside of the YAML block.

rules:
  - name: "Identified Pattern Name (e.g., Macro-Technical Divergence)"
    description: "A human-readable explanation of the pattern you found. e.g., 'When the 'macro_strategist' is bearish but 'technical_analyst' is bullish, the 'pnl_result' is consistently negative.'"
    action:
      type: "adjust_credibility"
      target_agent: "technical_analyst"
      adjustment_reason: "punitive_adjustment_on_bearish_macro"

**Logs to Analyze:**
{log_summary}

Please begin your YAML output now:
"""
        return prompt

    def _parse_yaml_rules(self, raw_yaml: str) -> List[Dict[str, Any]]:
        try:
            # Ensure we return the list of rules, even if nested under a 'rules' key
            parsed_data = yaml.safe_load(raw_yaml)
            if isinstance(parsed_data, dict) and 'rules' in parsed_data:
                return parsed_data.get('rules', [])
            elif isinstance(parsed_data, list):
                return parsed_data
            logger.warning(f"Could not parse rules from YAML output: {raw_yaml}")
            return []
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML rules: {e}")
            return []

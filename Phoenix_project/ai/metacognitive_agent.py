# ai/metacognitive_agent.py

import logging
import json
from typing import List, Dict, Any
from ai.tabular_db_client import TabularDBClient # Placeholder
from api.gemini_pool_manager import GeminiPoolManager, query_model # Placeholder
from observability import get_logger

logger = get_logger(__name__)

LLM_CLIENT = "gemini" # Assuming a default
class MetaCognitiveAgent:
    """
    L3 Agent: Performs meta-cognition by analyzing the system's own decision-making process.
    It runs periodically to analyze P&L results and decision logs.
    """
    def __init__(
        self,
        db_client: TabularDBClient,
        gemini_client: GeminiPoolManager,
        lookback_period: int = 30,
        min_logs_for_analysis: int = 50
    ):
        self.db_client = db_client
        self.gemini_client = gemini_client
        self.lookback_period = lookback_period
        self.min_logs_for_analysis = min_logs_for_analysis
        logger.info("MetaCognitiveAgent initialized.")

    async def run_periodic_analysis(self) -> List[Dict]:
        """
        Main entry point for the agent's periodic execution.
        """
        logger.info("L3 MetaCognitiveAgent running periodic analysis...")
        
        # --- Task 5.1: Implement Sliding Window Strategy ---
        # Fetch logs from a fixed, recent lookback period to manage cost and context.
        recent_logs = self.db_client.get_decision_logs(days_back=7)
        
        if len(recent_logs) < self.min_logs_for_analysis:
            logger.info(f"Not enough logs ({len(recent_logs)}) for meta-analysis. Minimum is {self.min_logs_for_analysis}. Skipping.")
            return []

        # 2. Build the prompt using the summarized logs
        prompt = await self._build_causal_inference_prompt(recent_logs)
        
        # 3. Call the high-tier model with the summarized prompt
        # This is expected to be a high-cost, high-capability model
        response_text = await self.gemini_client.query_model_async(prompt, model="gemini-1.5-pro-latest", client_name=LLM_CLIENT)
        
        # 3. Parse the response to extract actionable rules
        new_rules = self._parse_response_for_rules(response_text)
        
        return new_rules

    async def _summarize_logs(self, logs: List[Dict]) -> str:
        """
        Task 5.2: Pre-processes logs by summarizing them with a low-cost model.
        """
        logger.info(f"Summarizing {len(logs)} logs with a low-cost model...")
        
        log_texts = [json.dumps(log) for log in logs]
        
        # Simple chunking to avoid massive single prompts
        CHUNK_SIZE = 20
        summaries = []
        for i in range(0, len(log_texts), CHUNK_SIZE):
            chunk = "\n".join(log_texts[i:i+CHUNK_SIZE])
            prompt = f"""
            Analyze the following JSON log entries. Summarize the key patterns, decisions, and outcomes into 2-3 bullet points. Focus on relationships between agent evidence, fusion results, and P&L.
            --- LOGS ---
            {chunk}
            --- SUMMARY ---
            """
            # Use a low-cost model for this summarization task
            summary = await self.gemini_client.query_model_async(prompt, model="gemini-1.5-flash-latest", client_name=LLM_CLIENT)
            summaries.append(summary)
            
        return "\n".join(summaries)

    async def _build_causal_inference_prompt(self, logs: List[Dict]) -> str:
        """
        Constructs the prompt for the high-tier model, now using summarized logs.
        """
        # The prompt asks the LLM to perform causal inference.
        # The goal is to find causal links between agent behaviors, market conditions, and P&L.
        # For example: "The 'technical_analyst' agent consistently provides over-optimistic scores
        # during high-volatility periods, leading to negative P&L. A new rule should be injected..."
        
        summarized_logs = await self._summarize_logs(logs)
        
        prompt = f"""
        You are a Meta-Cognitive AI analyzing the performance of a multi-agent financial analysis system.
        Your task is to identify flawed reasoning patterns, biases, or incorrect causal links in the system's behavior.
        Analyze the following summaries of recent decision logs. Each summary represents a cluster of decisions and their outcomes.
        Based on these summaries, identify 1 to 3 systemic issues. For each issue, propose a specific, actionable rule to correct it.
        The rule MUST be in the format specified in the output schema.

        ### Summarized Decision Logs ###
        {summarized_logs}

        ### Causal Inference and Rule Generation ###
        Analyze the patterns and generate rules.
        Your output MUST be a valid JSON list containing rule objects.
        
        Example Output Format:
        [
          {{
            "name": "Add constraint for analyst bias",
            "action_type": "update_prompt",
            "target_agent": "fundamental_analyst",
            "description": "Always double-check revenue projections against 3rd-party data."
          }},
          {{
            "name": "Down-weight technicals in high-vol",
            "action_type": "adjust_credibility",
            "target_agent": "technical_analyst",
            "adjustment_factor": -0.2,
            "conditions": ["market_volatility > 0.8"]
          }}
        ]
        """
        return prompt

    def _parse_response_for_rules(self, response_text: str) -> List[Dict]:
        """
        Safely parses the LLM's JSON response into a list of rule dictionaries.
        """
        logger.info(f"Parsing L3 response for rules...")
        try:
            # Clean the response text (LLMs sometimes add markdown)
            if response_text.strip().startswith("```json"):
                response_text = response_text.strip()[7:-3].strip()
            
            rules = json.loads(response_text)
            if isinstance(rules, list):
                logger.info(f"Successfully parsed {len(rules)} new rules.")
                return rules
            else:
                logger.warning(f"L3 response was valid JSON but not a list: {type(rules)}")
                return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode L3 response JSON: {e}\nResponse was:\n{response_text}")
            return []
        except Exception as e:
            logger.error(f"Error parsing L3 rules: {e}", exc_info=True)
            return []

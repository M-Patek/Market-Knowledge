from typing import List, Dict, Any, Optional
from core.schemas.fusion_result import AgentDecision, FusionResult
from ai.prompt_manager import PromptManager
from api.gateway import APIGateway
from monitor.logging import get_logger

logger = get_logger(__name__)

class MetacognitiveAgent:
    """
    An agent that "thinks about thinking." It analyzes the reasoning
    of other agents, identifies potential biases or flaws, and provides
    a critique or a refined synthesis.
    
    This agent is a key part of the 'Arbitrator' or 'Critic' step.
    """

    def __init__(self, api_gateway: APIGateway, prompt_manager: PromptManager):
        self.api_gateway = api_gateway
        self.prompt_manager = prompt_manager
        logger.info("MetacognitiveAgent initialized.")

    async def critique_reasoning(
        self,
        context: str,
        agent_decisions: List[AgentDecision],
        prompt_name: str = "critique_reasoning"
    ) -> Dict[str, Any]:
        """
        Analyzes a list of agent decisions and provides a critique.
        
        Args:
            context (str): The shared context provided to all agents.
            agent_decisions (List[AgentDecision]): The outputs from the AI ensemble.
            prompt_name (str): The prompt template to use (e.g., "critique", "arbitrate").
            
        Returns:
            A dictionary containing the critique, identified biases, etc.
        """
        
        # Format the agent decisions for the prompt
        decisions_summary = ""
        for i, decision in enumerate(agent_decisions):
            decisions_summary += (
                f"--- Agent {i+1}: {decision.agent_name} (Model: {decision.model_id}) ---\n"
                f"Decision: {decision.decision} (Confidence: {decision.confidence:.2f})\n"
                f"Reasoning: {decision.reasoning}\n\n"
            )
            
        prompt = self.prompt_manager.get_prompt(
            prompt_name,
            context=context,
            decisions_summary=decisions_summary
        )
        
        if not prompt:
            logger.error(f"Could not get prompt '{prompt_name}'.")
            return {"error": f"Prompt '{prompt_name}' not found."}

        try:
            raw_response = await self.api_gateway.send_request(
                model_name="gemini-1.5-pro", # Use a highly capable model
                prompt=prompt,
                temperature=0.2,
                max_tokens=2048
            )
            
            # The response should ideally be structured (JSON)
            # For simplicity, we'll assume it's text, but JSON is better.
            critique = self._parse_critique_response(raw_response)
            logger.info("Metacognitive critique generated.")
            return critique
            
        except Exception as e:
            logger.error(f"Error during metacognitive critique: {e}", exc_info=True)
            return {"error": f"LLM API failed: {e}"}

    def _parse_critique_response(self, response: str) -> Dict[str, Any]:
        """
        Parses the raw response from the LLM.
        
        Assumes the prompt requested a JSON output like:
        {
            "summary_critique": "...",
            "identified_biases": ["..."],
            "conflicting_points": ["..."],
            "suggested_decision": "BUY/SELL/HOLD",
            "confidence": 0.0-1.0,
            "reasoning": "..."
        }
        """
        try:
            import json
            # Robust parsing (handles ```json ... ```)
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            else:
                json_str = response.strip()
                
            parsed_data = json.loads(json_str)
            return parsed_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON critique: {e}. Response: {response[:200]}...")
            # Fallback to text
            return {"summary_critique": response, "identified_biases": [], "error": "JSON parse failed"}
        except Exception as e:
            logger.error(f"Error parsing critique response: {e}", exc_info=True)
            return {"error": f"Parsing failed: {e}"}

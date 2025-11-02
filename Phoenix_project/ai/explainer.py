from typing import Dict, Any
from api.gateway import APIGateway
from ai.prompt_manager import PromptManager
from core.schemas.fusion_result import FusionResult
from monitor.logging import get_logger

logger = get_logger(__name__)

class Explainer:
    """
    Generates human-readable explanations for the final fused decision
    by synthesizing the reasoning of all contributing agents.
    """

    def __init__(self, api_gateway: APIGateway, prompt_manager: PromptManager):
        self.api_gateway = api_gateway
        self.prompt_manager = prompt_manager
        logger.info("Explainer initialized.")

    async def generate_explanation(self, fusion_result: FusionResult, context: str) -> str:
        """
        Generates a coherent explanation for the final decision.
        
        Args:
            fusion_result (FusionResult): The final output from the Synthesizer.
            context (str): The formatted context string that led to this decision.

        Returns:
            str: A human-readable explanation.
        """
        logger.debug(f"Generating explanation for decision: {fusion_result.final_decision}")

        # Collate reasoning from all agents
        contributing_reasoning = ""
        for agent_decision in fusion_result.contributing_agents:
            contributing_reasoning += (
                f"--- Agent: {agent_decision.agent_name} (Model: {agent_decision.model_id}) ---\n"
                f"Decision: {agent_decision.decision} (Confidence: {agent_decision.confidence:.2f})\n"
                f"Reasoning: {agent_decision.reasoning}\n\n"
            )
        
        prompt = self.prompt_manager.get_prompt(
            "explain_decision",
            context=context,
            final_decision=fusion_result.final_decision,
            final_confidence=fusion_result.confidence,
            synthesis_reasoning=fusion_result.reasoning,
            contributing_reasoning=contributing_reasoning
        )
        
        if not prompt:
            logger.error("Could not get 'explain_decision' prompt.")
            return "Error: Could not generate explanation (prompt missing)."

        try:
            explanation = await self.api_gateway.send_request(
                model_name="gemini-pro", # Use a capable model for synthesis
                prompt=prompt,
                temperature=0.3,
                max_tokens=1000
            )
            logger.info("Successfully generated explanation.")
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}", exc_info=True)
            return f"Error: Could not generate explanation ({e})."

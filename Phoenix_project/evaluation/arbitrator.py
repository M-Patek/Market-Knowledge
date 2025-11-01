import asyncio
from typing import Dict, Any, List

from ..monitor.logging import get_logger
from ..api.gemini_pool_manager import GeminiPoolManager
from ..ai.prompt_manager import PromptManager
from ..core.schemas.fusion_result import AgentDecision

logger = get_logger(__name__)

class Arbitrator:
    """
    An advanced LLM agent ("meta-agent") that is invoked when
    the primary agent ensemble shows significant disagreement or
    low confidence.
    
    Its job is to review the conflicting decisions and justifications
    and provide a final, arbitrated judgment.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        gemini_pool: GeminiPoolManager,
        prompt_manager: PromptManager
    ):
        """
        Initializes the Arbitrator.
        
        Args:
            config: Main configuration object.
            gemini_pool: Pool for accessing Gemini models.
            prompt_manager: To get the 'arbitrator' prompt.
        """
        self.config = config.get('arbitrator', {})
        self.gemini_pool = gemini_pool
        self.prompt_manager = prompt_manager
        
        # Use a powerful model for arbitration
        self.model_id = self.config.get('model_id', 'gemini-1.5-pro')
        
        logger.info(f"Arbitrator initialized with model: {self.model_id}")

    async def arbitrate(
        self,
        event_headline: str,
        conflicting_decisions: List[AgentDecision],
        context_summary: str,
        event_id: str
    ) -> AgentDecision:
        """
        Runs the arbitration process.
        
        Args:
            event_headline (str): The original event headline for context.
            conflicting_decisions (List[AgentDecision]): The list of decisions to arbitrate.
            context_summary (str): A compressed summary of the RAG context.
            event_id (str): The unique event ID for tracing.
            
        Returns:
            AgentDecision: The final, arbitrated decision.
        """
        
        # 1. Format the conflicting evidence for the prompt
        formatted_conflicts = self._format_conflicts(conflicting_decisions)
        
        # 2. Get prompts
        system_prompt = self.prompt_manager.get_system_prompt('arbitrator')
        # The 'arbitrator' user prompt template is assumed to take
        # 'headline', 'context_summary', and 'conflicts'.
        user_prompt_template = self.prompt_manager.get_user_prompt_template('arbitrator') 
        
        user_prompt = user_prompt_template.format(
            headline=event_headline,
            context_summary=context_summary,
            conflicts=formatted_conflicts
        )

        # 3. Call the LLM
        try:
            async with self.gemini_pool.get_client(self.model_id) as gemini_client:
                response = await gemini_client.generate_content_async(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    request_id=f"{event_id}_arbitrator",
                    generation_config={"response_mime_type": "application/json"}
                )
            
            # 4. Parse the response
            # The response is expected to match the AgentDecision schema
            return self._parse_arbitrator_response(response)

        except Exception as e:
            logger.error(f"Arbitrator failed to run for event {event_id}: {e}", exc_info=True)
            return AgentDecision(
                agent_id="arbitrator",
                decision="ERROR",
                confidence=0.0,
                justification=f"Arbitration LLM call failed: {str(e)}",
                metadata={"error": str(e)}
            )

    def _format_conflicts(self, decisions: List[AgentDecision]) -> str:
        """Helper to format the list of decisions into a string for the prompt."""
        formatted = ""
        for i, d in enumerate(decisions):
            if d.decision in ["ERROR", "INVALID_RESPONSE"]:
                continue
            
            formatted += f"--- Agent {i+1} ({d.agent_id}) ---\n"
            formatted += f"Decision: {d.decision}\n"
            formatted += f"Confidence: {d.confidence:.0%}\n"
            formatted += f"Justification: {d.justification}\n\n"
        
        return formatted

    def _parse_arbitrator_response(self, response: Dict[str, Any]) -> AgentDecision:
        """
D.
        """
        try:
            # Validate required keys
            if not all(k in response for k in ["decision", "confidence", "justification"]):
                raise KeyError(f"Arbitrator response missing one or more required keys.")

            confidence = float(response["confidence"])
            if not (0 <= confidence <= 1):
                raise ValueError(f"Confidence score {confidence} out of range [0, 1]")

            decision = AgentDecision(
                agent_id="arbitrator",
                decision=str(response["decision"]).upper(),
                confidence=confidence,
                justification=str(response["justification"]),
                metadata=response.get("metadata", {"arbitrated": True})
            )
            return decision

        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Failed to parse arbitrator response: {e}. Raw: {response}")
            return AgentDecision(
                agent_id="arbitrator",
                decision="INVALID_RESPONSE",
                confidence=0.0,
                justification=f"Failed to parse arbitration response: {str(e)}",
                metadata={"error": str(e), "raw_response": str(response)}
            )

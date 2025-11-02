"""
Metacognitive Agent.

This agent's role is not to analyze the primary event, but to analyze
the *outputs* of other agents. It reflects on the ensemble's decisions
to identify conflicts, assess overall confidence, and generate a
synthesizing rationale.
"""
import logging
from typing import List, Dict, Any, Optional

from ..api.gemini_pool_manager import GeminiPoolManager
from .prompt_manager import PromptManager
from .prompt_renderer import PromptRenderer
# 修复：使用正确的相对导入
from ..core.schemas.fusion_result import AgentDecision

logger = logging.getLogger(__name__)

class MetacognitiveAgent:
    """
    Analyzes a list of AgentDecisions to produce a summary and synthesis.
    
    This agent is often used by the Synthesizer or Arbitrator.
    """

    def __init__(self,
                 prompt_manager: PromptManager,
                 prompt_renderer: PromptRenderer,
                 gemini_pool: GeminiPoolManager,
                 prompt_name: str = "arbitrator"): # Default prompt
        
        self.prompt_manager = prompt_manager
        self.prompt_renderer = prompt_renderer
        self.gemini_pool = gemini_pool
        self.prompt_template = self.prompt_manager.get_prompt(prompt_name)
        
        if not self.prompt_template:
            raise ValueError(f"MetacognitiveAgent prompt '{prompt_name}' not found.")
            
        logger.info(f"MetacognitiveAgent initialized with prompt '{prompt_name}'.")

    async def synthesize(self, 
                         event_context: Dict[str, Any], 
                         agent_decisions: List[AgentDecision]) -> Dict[str, Any] | None:
        """
        Analyzes the decisions and generates a synthesized rationale.

        Args:
            event_context: The original event and evidence context.
            agent_decisions: The list of decisions from the first-layer agents.

        Returns:
            A dictionary containing the synthesized rationale and fused scores,
            or None if the analysis fails.
        """
        if not agent_decisions:
            logger.warning("MetacognitiveAgent: No agent decisions to synthesize.")
            return None

        # Convert Pydantic models to simple dicts for the prompt
        decisions_list_of_dicts = [d.dict() for d in agent_decisions]
        
        # Prepare the context for the renderer
        render_context = {
            "event": event_context.get("event", {}),
            "context": event_context.get("context", {}),
            "agent_decisions": decisions_list_of_dicts
        }

        try:
            # 1. Render the prompt
            full_prompt = self.prompt_renderer.render_from_context(
                template_content=self.prompt_template,
                render_context=render_context
            )
            
            # 2. Execute the LLM call
            # We expect the meta-agent to return a JSON object
            # matching the structure defined in its prompt (e.g., FusionResult fields)
            response_json = await self.gemini_pool.generate_json(
                model_name="gemini-1.5-pro", # Meta-cognition needs a strong model
                prompt=full_prompt,
                system_prompt="You are a senior investment strategist. Your role is to synthesize conflicting reports. Respond ONLY with the requested JSON schema."
            )
            
            if not response_json:
                logger.warning("MetacognitiveAgent did not return a valid JSON response.")
                return None

            logger.debug(f"MetacognitiveAgent synthesis: {response_json}")
            return response_json

        except Exception as e:
            logger.error(f"Error during MetacognitiveAgent synthesis: {e}", exc_info=True)
            return None

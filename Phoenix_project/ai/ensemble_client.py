from typing import Dict, Any, List, Optional
import asyncio
from core.schemas.fusion_result import AgentDecision  # This import will fail, needs to be fixed
from api.gateway import APIGateway
from ai.prompt_manager import PromptManager
from monitor.logging import get_logger

logger = get_logger(__name__)

class AIEnsembleClient:
    """
    Manages running multiple AI agents/models in parallel or sequence
    to generate a diverse set of opinions or analyses.
    """

    def __init__(self, api_gateway: APIGateway, prompt_manager: PromptManager, agent_configs: Dict[str, Any]):
        self.api_gateway = api_gateway
        self.prompt_manager = prompt_manager
        self.agent_configs = agent_configs  # Config for each agent in the ensemble
        logger.info(f"AIEnsembleClient initialized with {len(agent_configs)} agents.")

    async def run_ensemble(self, context: str, base_prompt_name: str, context_data: Dict[str, Any]) -> List[AgentDecision]:
        """
        Runs all configured agents in the ensemble against the given context.
        
        Args:
            context (str): The formatted context string for the prompt.
            base_prompt_name (str): The base name of the prompt (e.g., "analyst").
                                    Agents might use variants like "analyst_optimistic".
            context_data (Dict[str, Any]): Raw context data for agents that might need it.

        Returns:
            List[AgentDecision]: A list of decisions, one from each agent.
        """
        tasks = []
        for agent_name, config in self.agent_configs.items():
            prompt_name = config.get("prompt_name", base_prompt_name)
            model_id = config.get("model_id", "gemini-pro") # Default model
            
            task = self._run_agent(
                agent_name=agent_name,
                model_id=model_id,
                prompt_name=prompt_name,
                context=context,
                config=config
            )
            tasks.append(task)
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        decisions: List[AgentDecision] = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Agent execution failed: {result}", exc_info=True)
            elif result:
                decisions.append(result)
                
        logger.info(f"Ensemble run complete. Generated {len(decisions)} valid decisions.")
        return decisions

    async def _run_agent(self, agent_name: str, model_id: str, prompt_name: str, context: str, config: Dict[str, Any]) -> Optional[AgentDecision]:
        """
        Internal method to run a single agent and parse its output.
        """
        logger.debug(f"Running agent '{agent_name}' using model '{model_id}' and prompt '{prompt_name}'.")
        
        # Get prompt and apply agent-specific persona/instructions
        persona = config.get("persona", "You are a helpful assistant.")
        prompt = self.prompt_manager.get_prompt(
            prompt_name,
            context=context,
            persona=persona,
            # Pass other config items as potential template variables
            **config
        )
        
        if not prompt:
            logger.error(f"Could not get prompt '{prompt_name}' for agent '{agent_name}'.")
            return None

        try:
            raw_response = await self.api_gateway.send_request(
                model_name=model_id,
                prompt=prompt,
                temperature=config.get("temperature", 0.5),
                max_tokens=config.get("max_tokens", 1000),
                stop_sequences=config.get("stop_sequences", None)
            )
            
            return self._parse_agent_response(raw_response, agent_name, model_id)
            
        except Exception as e:
            logger.error(f"Error running agent '{agent_name}': {e}", exc_info=True)
            return None

    def _parse_agent_response(self, response: str, agent_name: str, model_id: str) -> Optional[AgentDecision]:
        """
        Parses the raw string response from an LLM into a structured AgentDecision.
        This needs to be robust, perhaps expecting JSON or a specific format.
        
        Placeholder implementation: Assumes a simple "DECISION: [BUY/SELL/HOLD]" format.
        A real implementation *must* expect a more structured output, like JSON.
        """
        logger.debug(f"Parsing response from '{agent_name}': {response[:100]}...")
        
        # --- This is a critical parsing step ---
        # In a robust system, the prompt would request JSON output,
        # and we would parse it here.
        
        # Example: Simple keyword parsing (Fragile!)
        decision_str = "HOLD" # Default
        if "BUY" in response.upper():
            decision_str = "BUY"
        elif "SELL" in response.upper():
            decision_str = "SELL"
        
        # Placeholder for confidence - this should come from the LLM
        confidence = 0.75 
        
        # Placeholder for reasoning - this should be the bulk of the response
        reasoning = response
        
        try:
            decision = AgentDecision(
                agent_name=agent_name,
                model_id=model_id,
                decision=decision_str,
                confidence=confidence,
                reasoning=reasoning,
                raw_response=response
            )
            return decision
        except Exception as e:
            # This could happen if AgentDecision validation fails
            logger.error(f"Failed to create AgentDecision for '{agent_name}': {e}", exc_info=True)
            return None

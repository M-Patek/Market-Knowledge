"""
Ensemble Client for managing parallel execution of multiple AI agents.

This client takes a common context (retrieved evidence), sends it to multiple
specialized AI agents (defined in prompts/ and agents/), and collects their
standardized outputs (AgentDecision).
"""
import logging
import asyncio
from typing import List, Dict, Any

from .prompt_manager import PromptManager
from .prompt_renderer import PromptRenderer
from ..api.gemini_pool_manager import GeminiPoolManager
# 修复：使用正确的相对导入
from ..core.schemas.fusion_result import AgentDecision

logger = logging.getLogger(__name__)

class EnsembleClient:
    """
    Manages concurrent API calls to a pool of AI agents.
    """

    def __init__(self, 
                 agent_registry_path: str,
                 prompt_manager: PromptManager,
                 prompt_renderer: PromptRenderer,
                 gemini_pool: GeminiPoolManager):
        """
        Initializes the EnsembleClient.

        Args:
            agent_registry_path: Path to the YAML file defining the agents.
            prompt_manager: Instance of PromptManager to load prompt templates.
            prompt_renderer: Instance of PromptRenderer to fill templates.
            gemini_pool: Instance of GeminiPoolManager to execute API calls.
        """
        self.agent_registry_path = agent_registry_path
        self.prompt_manager = prompt_manager
        self.prompt_renderer = prompt_renderer
        self.gemini_pool = gemini_pool
        self.agents = self._load_agent_registry()
        logger.info(f"EnsembleClient initialized with {len(self.agents)} agents.")

    def _load_agent_registry(self) -> Dict[str, Any]:
        """Loads agent definitions from the registry YAML."""
        # This would typically load a YAML file.
        # For simplicity, we'll use a mock registry if loading fails.
        try:
            # Placeholder: In a real app, use ConfigLoader
            # config = ConfigLoader.load_yaml(self.agent_registry_path)
            # return config.get('agents', {})
            
            # Mocking for this example:
            mock_agents = {
                "Analyst": {"model": "gemini-1.5-pro", "prompt_name": "analyst"},
                "FactChecker": {"model": "gemini-1.5-flash", "prompt_name": "fact_checker"},
                "ContextObserver": {"model": "gemini-1.5-flash", "prompt_name": "context_observer"}
            }
            logger.warning("Using mock agent registry. Implement YAML loading.")
            return mock_agents
            
        except Exception as e:
            logger.error(f"Failed to load agent registry from {self.agent_registry_path}: {e}", exc_info=True)
            return {}

    async def execute_ensemble(self, 
                               evidence_context: Dict[str, Any], 
                               event: Dict[str, Any]) -> List[AgentDecision]:
        """
        Executes all registered agents in parallel against the given context.

        Args:
            evidence_context: The dictionary of retrieved evidence (RAG context).
            event: The triggering event (e.g., a MarketEvent as a dict).

        Returns:
            A list of AgentDecision objects, one for each successful agent response.
        """
        tasks = []
        for agent_name, agent_config in self.agents.items():
            task = self._run_agent(
                agent_name=agent_name,
                agent_config=agent_config,
                evidence_context=evidence_context,
                event=event
            )
            tasks.append(task)
            
        results = await asyncio.gather(*tasks)
        
        # Filter out failed tasks (which return None)
        successful_decisions = [res for res in results if res is not None]
        
        logger.info(f"Ensemble execution complete. Received {len(successful_decisions)} decisions out of {len(self.agents)} agents.")
        return successful_decisions

    async def _run_agent(self, 
                         agent_name: str, 
                         agent_config: Dict[str, Any], 
                         evidence_context: Dict[str, Any], 
                         event: Dict[str, Any]) -> AgentDecision | None:
        """
        Prepares, executes, and parses the output for a single agent.
        """
        try:
            # 1. Get the prompt template
            prompt_name = agent_config.get("prompt_name")
            prompt_template = self.prompt_manager.get_prompt(prompt_name)
            if not prompt_template:
                logger.error(f"Agent {agent_name}: Prompt template '{prompt_name}' not found.")
                return None

            # 2. Render the prompt
            full_prompt = self.prompt_renderer.render(
                template_content=prompt_template,
                context=evidence_context,
                event=event
            )
            
            # 3. Execute the LLM call
            model_name = agent_config.get("model", "gemini-1.5-pro")
            
            # We expect the agent to return a JSON object matching AgentDecision
            response_json = await self.gemini_pool.generate_json(
                model_name=model_name,
                prompt=full_prompt,
                system_prompt="You are a financial expert. Respond ONLY with the requested JSON schema."
            )
            
            if not response_json:
                logger.warning(f"Agent {agent_name} did not return a valid JSON response.")
                return None

            # 4. Parse and validate the response
            # Add agent_name to the response data before validation
            response_json['agent_name'] = agent_name 
            
            # 修复：使用 AgentDecision schema
            decision = AgentDecision(**response_json)
            logger.debug(f"Agent {agent_name} decision: {decision.sentiment}, {decision.confidence}")
            return decision

        except Exception as e:
            logger.error(f"Error running agent {agent_name}: {e}", exc_info=True)
            return None

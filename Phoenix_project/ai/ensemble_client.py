import json
import asyncio
from typing import List, Dict, Any, Tuple

from .prompt_renderer import PromptRenderer
from ..api.gemini_pool_manager import GeminiPoolManager
from ..core.schemas.fusion_result import AgentDecision, AgentIO

class EnsembleClient:
    """
    Manages the execution of multiple AI agents (the "ensemble") in parallel.
    It formats requests, sends them to the GeminiPoolManager, and parses the structured
    JSON responses into AgentDecision objects.
    """

    def __init__(self, config: Dict[str, Any], gemini_pool: GeminiPoolManager):
        self.config = config
        self.gemini_pool = gemini_pool
        self.prompt_renderer = PromptRenderer()
        # Defines the agents that will run in the ensemble.
        # The 'role' must match a prompt template name (e.g., 'analyst', 'fact_checker').
        self.ensemble_definition = config.get('ai_ensemble', {}).get('agents', [])

    async def run_ensemble(
        self,
        context_bundle: Dict[str, Any],
        system_instructions: Dict[str, str],
        event_id: str
    ) -> Tuple[List[AgentDecision], Dict[str, AgentIO]]:
        """
        Executes the full agent ensemble concurrently for a given context bundle.

        Args:
            context_bundle: The dictionary containing all RAG-retrieved context
                            (e.g., {"vector_context": "...", "temporal_context": "..."}).
            system_instructions: A dictionary mapping agent roles to their system prompts.
            event_id: The unique ID for this processing event, used for logging and API calls.

        Returns:
            A tuple containing:
            1. A list of AgentDecision objects.
            2. A dictionary of AgentIO objects (raw inputs/outputs) for auditing.
        """
        tasks = []
        
        # Create a task for each agent defined in the configuration
        for agent_def in self.ensemble_definition:
            role = agent_def.get('role')
            model_id = agent_def.get('model_id')
            
            if not role or not model_id:
                continue

            system_prompt = system_instructions.get(role)
            if not system_prompt:
                continue

            # Render the user prompt for this specific agent
            user_prompt = self.prompt_renderer.render_prompt(role, context_bundle)

            # Create an asynchronous task for the API call
            task = asyncio.create_task(
                self._execute_agent_call(
                    role=role,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model_id=model_id,
                    event_id=event_id,
                )
            )
            tasks.append((role, task))

        # Wait for all tasks to complete
        results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

        # Process results
        agent_decisions: List[AgentDecision] = []
        agent_io_map: Dict[str, AgentIO] = {}

        for (role, task), result in zip(tasks, results):
            if isinstance(result, Exception):
                # Handle exceptions during the API call
                decision = AgentDecision(
                    agent_id=role,
                    decision="ERROR",
                    confidence=0.0,
                    justification=f"Agent execution failed: {str(result)}",
                    metadata={"error": str(result)}
                )
                raw_response = f"AGENT_EXECUTION_ERROR: {str(result)}"
            else:
                # Parse the successful response
                decision, raw_response = self._parse_agent_response(role, result)
            
            agent_decisions.append(decision)
            
            # Store raw I/O for auditing
            # Note: Storing the full user_prompt (which includes all context)
            # can be very large. Consider truncating or summarizing if needed.
            agent_io_map[role] = AgentIO(
                system_prompt=system_instructions.get(role, ""),
                user_prompt=self.prompt_renderer.render_prompt(role, context_bundle), # Re-render for audit log
                raw_response=raw_response,
                parsed_decision=decision
            )

        return agent_decisions, agent_io_map

    async def _execute_agent_call(
        self,
        role: str,
        system_prompt: str,
        user_prompt: str,
        model_id: str,
        event_id: str
    ) -> Dict[str, Any]:
        """
        Internal method to acquire a Gemini client from the pool and make the API call.
        """
        # Acquire a client from the resource pool
        async with self.gemini_pool.get_client(model_id) as gemini_client:
            # Generate the content
            response = await gemini_client.generate_content_async(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                request_id=f"{event_id}_{role}",
                generation_config={"response_mime_type": "application/json"}
            )
            return response # This is assumed to be the parsed JSON dictionary

    def _parse_agent_response(self, agent_id: str, response: Dict[str, Any]) -> Tuple[AgentDecision, str]:
        """
        Parses the structured JSON response from the LLM into an AgentDecision.
        Includes robust error handling for malformed JSON or missing keys.
        """
        raw_response_str = json.dumps(response)
        try:
            # 'response' is already the parsed JSON object from the client
            decision_data = response

            # Validate required keys
            if not all(k in decision_data for k in ["decision", "confidence", "justification"]):
                raise KeyError(f"Missing one or more required keys: decision, confidence, justification")

            # Perform type conversion and validation
            confidence = float(decision_data["confidence"])
            if not (0 <= confidence <= 1):
                raise ValueError(f"Confidence score {confidence} out of range [0, 1]")

            decision = AgentDecision(
                agent_id=agent_id,
                decision=str(decision_data["decision"]).upper(),
                confidence=confidence,
                justification=str(decision_data["justification"]),
                metadata=decision_data.get("metadata", {})
            )
            return decision, raw_response_str

        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            # Handle any parsing, key, or type error
            error_message = f"Failed to parse agent response: {e}. Raw: {raw_response_str[:500]}..."
            
            decision = AgentDecision(
                agent_id=agent_id,
                decision="INVALID_RESPONSE",
                confidence=0.0,
                justification=error_message,
                metadata={"error": str(e), "raw_response": raw_response_str}
            )
            return decision, raw_response_str

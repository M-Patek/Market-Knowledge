from typing import List, Dict, Any
from ai.prompt_manager import PromptManager
from api.gateway import APIGateway
from ai.metacognitive_agent import MetacognitiveAgent
from core.schemas.fusion_result import AgentDecision
from monitor.logging import get_logger

logger = get_logger(__name__)

class Arbitrator:
    """
    A specialized evaluation agent that resolves conflicts between
    the decisions of other AI agents.
    
    It uses a MetacognitiveAgent to perform the core analysis.
    """

    def __init__(
        self,
        api_gateway: APIGateway,
        prompt_manager: PromptManager,
        metacognitive_agent: MetacognitiveAgent
    ):
        self.api_gateway = api_gateway
        self.prompt_manager = prompt_manager
        self.metacognitive_agent = metacognitive_agent
        self.prompt_name = "arbitrate_decisions" # Specific prompt
        logger.info("Arbitrator initialized.")

    async def arbitrate(self, context: str, agent_decisions: List[AgentDecision]) -> Dict[str, Any]:
        """
        Analyzes conflicting agent decisions and suggests a final,
        arbitrated decision.
        
        Args:
            context (str): The shared context provided to all agents.
            agent_decisions (List[AgentDecision]): The outputs from the AI ensemble.

        Returns:
            Dict[str, Any]: A structured response, e.g.:
            {
                "summary_critique": "...",
                "identified_biases": ["..."],
                "suggested_decision": "BUY/SELL/HOLD",
                "confidence": 0.0-1.0,
                "reasoning": "..."
            }
        """
        
        # Check if there is actual conflict
        decisions = {d.decision for d in agent_decisions}
        if len(decisions) <= 1:
            logger.info("Arbitrator: No conflict found. Skipping full arbitration.")
            # Return a simple "no conflict" response
            first_decision = agent_decisions[0]
            return {
                "summary_critique": "No significant conflict detected among agents.",
                "identified_biases": [],
                "suggested_decision": first_decision.decision,
                "confidence": first_decision.confidence,
                "reasoning": "Consensus was achieved by the ensemble."
            }

        logger.info(f"Arbitrator: Conflict detected ({decisions}). Running metacognitive analysis.")
        
        # Use the metacognitive agent with the "arbitrate_decisions" prompt
        try:
            arbitration_result = await self.metacognitive_agent.critique_reasoning(
                context=context,
                agent_decisions=agent_decisions,
                prompt_name=self.prompt_name
            )
            return arbitration_result
            
        except Exception as e:
            logger.error(f"Error during arbitration: {e}", exc_info=True)
            return {
                "summary_critique": "Arbitration failed due to an internal error.",
                "identified_biases": [],
                "suggested_decision": "ERROR_HOLD",
                "confidence": 0.0,
                "reasoning": f"Arbitration failed: {e}"
            }

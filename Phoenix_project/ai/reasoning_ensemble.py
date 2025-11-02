from typing import Dict, Any, List
import asyncio
from api.gateway import APIGateway
from ai.prompt_manager import PromptManager
from ai.data_adapter import DataAdapter
from ai.ensemble_client import AIEnsembleClient
from ai.metacognitive_agent import MetacognitiveAgent
from fusion.synthesizer import Synthesizer
from evaluation.critic import Critic
from evaluation.arbitrator import Arbitrator
from core.schemas.fusion_result import FusionResult, AgentDecision
from core.pipeline_state import PipelineState
from monitor.logging import get_logger

logger = get_logger(__name__)

class ReasoningEnsemble:
    """
    Coordinates the entire AI reasoning pipeline:
    1. Data Adapting: Formats context data for LLMs.
    2. Ensemble Run: Runs multiple AI agents in parallel.
    3. Arbitration/Critique: (Optional) A metacognitive agent analyzes the results.
    4. Synthesis: Fuses the agent decisions into a single, final result.
    """

    def __init__(
        self,
        api_gateway: APIGateway,
        prompt_manager: PromptManager,
        config: Dict[str, Any]
    ):
        self.api_gateway = api_gateway
        self.prompt_manager = prompt_manager
        self.config = config
        
        # 1. Data Adapter
        self.data_adapter = DataAdapter(config.get("data_adapter", {}))
        
        # 2. Ensemble Client
        ensemble_agent_configs = config.get("ensemble_agents", {})
        if not ensemble_agent_configs:
            logger.warning("No ensemble agents configured. Reasoning will be limited.")
        self.ensemble_client = AIEnsembleClient(
            api_gateway, prompt_manager, ensemble_agent_configs
        )
        
        # 3. Metacognitive / Evaluation Agents
        self.metacognitive_agent = MetacognitiveAgent(api_gateway, prompt_manager)
        self.arbitrator = Arbitrator(
            api_gateway, prompt_manager, self.metacognitive_agent
        )
        self.critic = Critic(api_gateway, prompt_manager) # General purpose critic
        
        # 4. Synthesizer
        self.synthesizer = Synthesizer(
            api_gateway, prompt_manager, config.get("synthesizer", {})
        )
        
        self.use_arbitrator = config.get("use_arbitrator", True)
        logger.info("ReasoningEnsemble initialized.")

    async def reason(self, pipeline_state: PipelineState) -> FusionResult:
        """
        Executes the full reasoning pipeline.
        
        Args:
            pipeline_state (PipelineState): The current state of the system.
            
        Returns:
            FusionResult: The final, synthesized decision.
        """
        logger.info("Starting reasoning pipeline...")
        
        # 1. Get and format context
        context_data = pipeline_state.get_full_context()
        try:
            formatted_context = self.data_adapter.format_context(context_data)
            logger.debug(f"Formatted context: \n{formatted_context[:300]}...")
        except Exception as e:
            logger.error(f"Error formatting context: {e}", exc_info=True)
            return FusionResult(
                final_decision="ERROR",
                confidence=0.0,
                reasoning=f"Failed to format context: {e}",
                contributing_agents=[]
            )

        # 2. Run AI Ensemble
        try:
            agent_decisions = await self.ensemble_client.run_ensemble(
                context=formatted_context,
                base_prompt_name="analyst_persona", # Base prompt
                context_data=context_data
            )
        except Exception as e:
            logger.error(f"Error running AI ensemble: {e}", exc_info=True)
            return FusionResult(
                final_decision="ERROR",
                confidence=0.0,
                reasoning=f"AI ensemble failed: {e}",
                contributing_agents=[]
            )
            
        if not agent_decisions:
            logger.warning("AI ensemble returned no decisions.")
            return FusionResult(
                final_decision="HOLD",
                confidence=0.5,
                reasoning="AI ensemble provided no opinions. Defaulting to HOLD.",
                contributing_agents=[]
            )
            
        pipeline_state.update_value("last_agent_decisions", agent_decisions)

        # 3. Arbitration / Critique (Metacognition)
        arbitration_result = None
        if self.use_arbitrator:
            try:
                logger.info("Running Arbitrator...")
                arbitration_result = await self.arbitrator.arbitrate(
                    formatted_context, agent_decisions
                )
                pipeline_state.update_value("last_arbitration", arbitration_result)
                logger.info(f"Arbitrator suggestion: {arbitration_result.get('suggested_decision')}")
            except Exception as e:
                logger.error(f"Error running Arbitrator: {e}", exc_info=True)
                # Non-fatal, we can proceed to synthesis without it
        
        # 4. Synthesis
        try:
            logger.info("Running Synthesizer...")
            final_result = await self.synthesizer.fuse_decisions(
                context=formatted_context,
                agent_decisions=agent_decisions,
                arbitrator_critique=arbitration_result # Pass critique if available
            )
        except Exception as e:
            logger.error(f"Error running Synthesizer: {e}", exc_info=True)
            return FusionResult(
                final_decision="ERROR",
                confidence=0.0,
                reasoning=f"Synthesizer failed: {e}",
                contributing_agents=agent_decisions
            )
            
        logger.info(f"Reasoning pipeline complete. Final decision: {final_result.final_decision}")
        pipeline_state.update_value("last_fusion_result", final_result)
        
        return final_result

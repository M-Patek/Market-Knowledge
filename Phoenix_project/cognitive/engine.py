import asyncio
from typing import Dict, Any, Optional

# 修复：全部改为相对导入
from ..ai.retriever import Retriever
from ..ai.reasoning_ensemble import ReasoningEnsemble
from ..evaluation.voter import Voter
from ..evaluation.critic import Critic
from ..evaluation.arbitrator import Arbitrator
from ..fusion.synthesizer import Synthesizer
from ..core.pipeline_state import PipelineState
from ..agents.executor import AgentExecutor
from ..monitor.logging import get_logger

logger = get_logger(__name__)

class CognitiveEngine:
    """
    The core cognitive pipeline (L1-L3).
    This engine integrates retrieval, agentic execution, reasoning,
    evaluation, and synthesis.
    """
    
    def __init__(self, 
                 retriever: Retriever, 
                 agent_executor: AgentExecutor,
                 reasoning_ensemble: ReasoningEnsemble,
                 voter: Voter,
                 critic: Critic,
                 arbitrator: Arbitrator,
                 synthesizer: Synthesizer,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the cognitive engine with all its components.
        """
        self.retriever = retriever
        self.agent_executor = agent_executor
        self.reasoning_ensemble = reasoning_ensemble
        self.voter = voter
        self.critic = critic
        self.arbitrator = arbitrator
        self.synthesizer = synthesizer
        self.config = config or {}
        logger.info("CognitiveEngine initialized.")

    async def run_pipeline(self, initial_state: PipelineState) -> PipelineState:
        """
        Executes the full L1-L3 cognitive pipeline.
        
        Args:
            initial_state (PipelineState): The starting state, usually containing
                                           the query and time range.

        Returns:
            PipelineState: The final, synthesized state after the pipeline run.
        """
        logger.info(f"Cognitive pipeline starting for query: {initial_state.query}")
        
        current_state = initial_state.model_copy(deep=True)

        try:
            # --- L1: RETRIEVAL & AGENTS ---
            logger.debug("Running L1: Retrieval and Agents")
            
            # L1a: Retrieval (Hybrid RAG)
            retrieved_context = await self.retriever.retrieve(
                current_state.query, 
                current_state.start_time, 
                current_state.end_time
            )
            current_state.retrieved_context = retrieved_context
            
            # L1b: Agentic Execution
            # Agents run in parallel, using the retrieved context
            agent_results = await self.agent_executor.run_agents(
                retrieved_context, 
                current_state
            )
            current_state.agent_outputs = agent_results

            # --- L2: REASONING & EVALUATION ---
            logger.debug("Running L2: Reasoning and Evaluation")

            # L2a: Reasoning Ensemble
            # Heterogeneous reasoners (Symbolic, LLM) run in parallel
            reasoning_outputs = await self.reasoning_ensemble.run(
                current_state, 
                {"retrieved_context": retrieved_context, "agent_results": agent_results}
            )
            current_state.reasoning_outputs = reasoning_outputs
            
            # L2b: Evaluation (Vote & Critique)
            # Voter aggregates outputs
            votes = self.voter.vote(reasoning_outputs, agent_results)
            current_state.votes = votes

            # Critic reviews the votes and outputs for conflicts/biases
            critiques = self.critic.review(current_state, votes)
            current_state.critiques = critiques

            # --- L3: FUSION & SYNTHESIS ---
            logger.debug("Running L3: Fusion and Synthesis")

            # L3a: Arbitrator resolves conflicts based on critiques
            final_decision = self.arbitrator.resolve(critiques, votes)
            current_state.arbitrator_decision = final_decision
            
            # L3b: Synthesizer fuses the final decision into a coherent insight
            final_state = self.synthesizer.fuse(current_state, final_decision)
            
            logger.info(f"Cognitive pipeline finished. Final insight: {final_state.fusion_result.insight}")
            return final_state

        except Exception as e:
            logger.error(f"Error during cognitive pipeline execution: {e}", exc_info=True)
            # Return the state as it was when the error occurred
            current_state.error_message = str(e)
            return current_state


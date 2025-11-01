import asyncio
from typing import Dict, Any, List

# 修复：全部改为相对导入
from ..agents.executor import AgentExecutor
from ..ai.retriever import Retriever
from ..evaluation.voter import Voter
from ..evaluation.critic import Critic
from ..evaluation.arbitrator import Arbitrator
from ..fusion.synthesizer import Synthesizer
from ..core.pipeline_state import PipelineState
from ..context_bus import ContextBus
from ..registry import Registry
from ..monitor.logging import get_logger

logger = get_logger(__name__)

class Orchestrator:
    """
    The main orchestrator component that manages the high-level
    workflow of the cognitive pipeline.
    """
    
    def __init__(self, 
                 agent_executor: AgentExecutor, 
                 retriever: Retriever, 
                 voter: Voter,
                 critic: Critic,
                 arbitrator: Arbitrator,
                 synthesizer: Synthesizer,
                 context_bus: ContextBus,
                 registry: Registry):
        """
        Initializes the Orchestrator with all necessary components.
        """
        self.agent_executor = agent_executor
        self.retriever = retriever
        self.voter = voter
        self.critic = critic
        self.arbitrator = arbitrator
        self.synthesizer = synthesizer
        self.context_bus = context_bus
        self.registry = registry # Component registry
        logger.info("Orchestrator initialized.")

    async def build_graph(self, query: str) -> Dict[str, Any]:
        """
        (Placeholder) Dynamically builds an execution graph or plan 
        based on the query.
        
        In a real implementation, this might involve an LLM call
        to decompose the query into sub-tasks.
        """
        logger.debug(f"Building execution graph for query: {query}")
        # Simplified plan: just run the standard agent set
        plan = {
            "steps": [
                {"name": "L1_Retrieve", "type": "retrieval"},
                {"name": "L1_Execute_Agents", "type": "agent_execution", "agents": "all"},
                {"name": "L2_Reason", "type": "reasoning"},
                {"name": "L2_Evaluate", "type": "evaluation"},
                {"name": "L3_Fuse", "type": "synthesis"}
            ]
        }
        return plan

    async def run_pipeline(self, query: str, config: Dict[str, Any]) -> PipelineState:
        """
        Runs the full end-to-end cognitive pipeline for a given query.
        
        Args:
            query (str): The user or system query (e.g., "Analyze TSLA outlook").
            config (Dict[str, Any]): Runtime configuration, including start/end times.

        Returns:
            PipelineState: The final state containing the synthesized insight.
        """
        logger.info(f"Orchestrator starting pipeline for query: {query}")
        
        try:
            # 1. Plan
            plan = await self.build_graph(query)
            
            # 2. Initialize State
            # PipelineState holds all data for this run
            state = PipelineState(query=query, graph=plan, **config)
            
            # 3. Retrieve (L1)
            # Use hybrid RAG to gather context
            context = await self.retriever.retrieve(
                query=state.query,
                start_date=state.start_time,
                end_date=state.end_time
            )
            state.retrieved_context = context
            
            # 4. Execute Agents (L1)
            # Agents process context in parallel
            agent_outputs = await self.agent_executor.run_agents(
                context=context,
                state=state
            )
            state.agent_outputs = agent_outputs
            
            # 5. Evaluate (L2)
            # Voter aggregates agent outputs
            votes = self.voter.vote(agent_outputs)
            state.votes = votes
            
            # Critic reviews for conflicts
            critiques = self.critic.review(state, votes)
            state.critiques = critiques
            
            # 6. Resolve (L3)
            # Arbitrator makes a final call based on critiques
            consensus = self.arbitrator.resolve(critiques, votes)
            state.arbitrator_decision = consensus
            
            # 7. Synthesize (L3)
            # Synthesizer generates the final human-readable insight
            final_state = self.synthesizer.fuse(state, consensus)
            
            # 8. Publish
            # Send the final result to the context bus for other systems
            await self.context_bus.publish("system_insights", final_state.fusion_result.model_dump())
            
            logger.info(f"Orchestrator finished pipeline. Final insight: {final_state.fusion_result.insight}")
            return final_state

        except Exception as e:
            logger.error(f"Critical error in orchestrator pipeline: {e}", exc_info=True)
            # Create a failed state
            failed_state = PipelineState(query=query, graph={}, **config)
            failed_state.error_message = str(e)
            return failed_state


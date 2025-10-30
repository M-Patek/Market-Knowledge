import asyncio
import logging
from typing import List, Dict, Any
from ai.agent_registry import AgentRegistry
from audit_manager import AuditManager  # (L2) Import the AuditManager

logger = logging.getLogger(__name__)

class L1_Task_Orchestrator:
    """
    (L2) Manages the asynchronous execution of multiple L1 agents.
    """
    def __init__(self, agent_registry: AgentRegistry, audit_manager: AuditManager):
        self.agent_registry = agent_registry
        self.audit_manager = audit_manager  # (L2) Store the AuditManager instance
        logger.info("L1_Task_Orchestrator initialized.")

    async def _run_single_agent_async(self, agent_name: str, context: str) -> Dict[str, Any]:
        """
        (Placeholder) Simulates a single async LLM call for an L1 agent.
        In a real implementation, this would involve:
        1. Getting agent config from the registry.
        2. Loading the prompt.
        3. Making an async API call to the LLM.
        4. Returning the structured result.
        """
        logger.info(f"Starting execution for L1 agent: '{agent_name}'...")
        # Simulate network latency of an API call
        await asyncio.sleep(1.5) 
        
        # Mocked result
        result = {
            "agent_name": agent_name,
            "analysis": f"Mock analysis from {agent_name} based on context: {context}",
            "confidence": 0.85
        }
        logger.info(f"Finished execution for L1 agent: '{agent_name}'.")
        return result

    async def run_all_agents_async(self, context: str, decision_id: str) -> List[Dict[str, Any]]:
        """
        (L2) Executes all registered L1 agents concurrently using asyncio.gather.

        Args:
            context: The shared context/prompt data to be sent to all agents.
            decision_id: The unique ID for this decision process, for auditing.

        Returns:
            A list of EvidenceItem-like dictionaries from all agents.
        """
        agent_names = self.agent_registry.list_agents()
        if not agent_names:
            logger.warning("No L1 agents registered. Returning empty list.")
            return []
        
        logger.info(f"Creating tasks for {len(agent_names)} L1 agents...")
        tasks = [
            self._run_single_agent_async(name, context) for name in agent_names
        ]
        
        results = await asyncio.gather(*tasks)
        logger.info(f"All L1 agent tasks have completed for decision_id: {decision_id}.")
        
        # (L2) Log the results to the AuditManager
        # (L7) We assume model_versions is passed into here or fetched
        mock_model_versions = {name: "1.0.0" for name in agent_names}
        self.audit_manager.log_l1_evidence(decision_id, results, mock_model_versions)
        
        return results

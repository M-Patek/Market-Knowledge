import asyncio
from typing import Dict, Any, Callable
from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

class AgentExecutor:
    """
    Handles the execution of a DAG (Directed Acyclic Graph) of agents.
    It resolves dependencies and executes agents in the correct order,
    handling both synchronous and asynchronous agent 'run' methods.
    """
    
    def __init__(self, agent_registry: Dict[str, Callable[[], Any]]):
        """
        Initializes the executor.
        
        Args:
            agent_registry (Dict[str, Callable[[], Any]]): 
                A dictionary mapping agent IDs to factory functions that
                create an instance of the agent.
        """
        self.agent_registry = agent_registry
        self.agent_instances = {} # Cache for instantiated agents
        logger.info(f"AgentExecutor initialized with {len(agent_registry)} agents.")

    def _get_agent_instance(self, agent_id: str) -> Any:
        """
        Retrieves or creates an agent instance.
        """
        if agent_id not in self.agent_instances:
            if agent_id not in self.agent_registry:
                logger.error(f"Agent '{agent_id}' not found in registry.")
                raise ValueError(f"Agent '{agent_id}' not found in registry.")
            
            # Call the factory function to create the agent
            try:
                self.agent_instances[agent_id] = self.agent_registry[agent_id]()
                logger.debug(f"Instantiated agent: {agent_id}")
            except Exception as e:
                logger.error(f"Failed to instantiate agent '{agent_id}': {e}", exc_info=True)
                raise
        
        return self.agent_instances[agent_id]

    async def execute_dag(
        self, 
        dag: Dict[str, List[str]], 
        state: PipelineState
    ) -> Dict[str, Any]:
        """
        Executes a cognitive DAG.
        
        Args:
            dag (Dict[str, List[str]]): 
                The dependency graph. 
                Keys are agent IDs (tasks), values are lists of agent IDs 
                they depend on.
            state (PipelineState): The current pipeline state.
            
        Returns:
            Dict[str, Any]: A dictionary mapping agent IDs to their results.
        """
        logger.info(f"Executing DAG with {len(dag)} tasks...")
        
        # Simple topological sort (assumes no circular dependencies)
        # In a real system, we'd validate the DAG first.
        
        execution_order = self._resolve_execution_order(dag)
        results: Dict[str, Any] = {}
        
        for agent_id in execution_order:
            if agent_id not in dag:
                logger.error(f"Task '{agent_id}' is in execution order but not in DAG definition.")
                continue
                
            dependencies = dag[agent_id]
            
            # Gather dependency results
            try:
                dependency_outputs = {dep_id: results[dep_id] for dep_id in dependencies}
            except KeyError as e:
                logger.error(f"Missing dependency result for '{agent_id}': {e}", exc_info=True)
                raise ValueError(f"Failed to execute '{agent_id}': Missing dependency {e}")
            
            try:
                # Get the agent instance
                agent = self._get_agent_instance(agent_id)
                
                logger.debug(f"Running agent: {agent_id}...")
                
                # Execute the agent's run method
                if asyncio.iscoroutinefunction(agent.run):
                    result = await agent.run(state, dependency_outputs)
                else:
                    result = agent.run(state, dependency_outputs)
                    
                results[agent_id] = result
                logger.debug(f"Agent {agent_id} complete.")
                
            except Exception as e:
                logger.error(f"Agent '{agent_id}' failed during execution: {e}", exc_info=True)
                # Propagate the error. The Orchestrator will handle it.
                raise
        
        logger.info("DAG execution complete.")
        return results

    def _resolve_execution_order(self, dag: Dict[str, List[str]]) -> List[str]:
        """
        Performs a topological sort on the DAG.
        (Simple implementation)
        """
        # 1. Find in-degrees (how many nodes point to me)
        in_degree = {agent_id: 0 for agent_id in dag}
        adj = {agent_id: [] for agent_id in dag}
        
        for agent_id, dependencies in dag.items():
            for dep_id in dependencies:
                # dep_id -> agent_id
                if dep_id in adj:
                    adj[dep_id].append(agent_id)
                if agent_id in in_degree:
                    in_degree[agent_id] += 1
                
        # 2. Find nodes with 0 in-degree
        queue = [agent_id for agent_id in dag if in_degree[agent_id] == 0]
        
        sorted_order = []
        
        # 3. Process the queue
        while queue:
            agent_id = queue.pop(0)
            sorted_order.append(agent_id)
            
            # "Remove" this node by decrementing its neighbors' in-degrees
            for neighbor in adj.get(agent_id, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
                    
        # 4. Check for cycles
        if len(sorted_order) != len(dag):
            cycle_nodes = {agent_id for agent_id, degree in in_degree.items() if degree > 0}
            logger.error(f"Cycle detected in agent DAG. Nodes involved: {cycle_nodes}")
            raiseValueError(f"Cycle detected in agent DAG. Nodes: {cycle_nodes}")
            
        return sorted_order

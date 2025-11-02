import asyncio
from typing import Dict, Any, Callable
from core.pipeline_state import PipelineState
from monitor.logging import get_logger

logger = get_logger(__name__)

class AgentExecutor:
    """
    Executes registered agents (functions or callables) based on triggers
    and updates the pipeline state.
    """

    def __init__(self, pipeline_state: PipelineState):
        self.pipeline_state = pipeline_state
        self.agents: Dict[str, Callable] = {}
        self.triggers: Dict[str, str] = {}  # Map agent name to trigger event
        logger.info("AgentExecutor initialized.")

    def register_agent(self, name: str, agent_callable: Callable, trigger_event: str):
        """
        Register an agent function or method to be called on a specific trigger.

        Args:
            name (str): A unique name for the agent.
            agent_callable (Callable): The function/method to call.
                                       Expected signature: agent(pipeline_state, **kwargs) -> update_dict
            trigger_event (str): The event name that triggers this agent.
        """
        if name in self.agents:
            logger.warning(f"Agent '{name}' is already registered. Overwriting.")
        self.agents[name] = agent_callable
        self.triggers[name] = trigger_event
        logger.info(f"Agent '{name}' registered with trigger '{trigger_event}'.")

    async def on_event(self, event_name: str, **kwargs):
        """
        Called by the EventDistributor when a relevant event occurs.
        This method finds and executes all agents registered for this event.
        """
        logger.debug(f"Event '{event_name}' received. Checking for triggered agents.")
        agents_to_run = [
            (name, agent)
            for name, trigger in self.triggers.items()
            if trigger == event_name
        ]

        if not agents_to_run:
            logger.debug(f"No agents registered for event '{event_name}'.")
            return

        for name, agent_callable in agents_to_run:
            logger.info(f"Executing agent '{name}' triggered by '{event_name}'.")
            try:
                # Asynchronously execute the agent
                if asyncio.iscoroutinefunction(agent_callable):
                    update_data = await agent_callable(
                        self.pipeline_state, **kwargs
                    )
                else:
                    # Run synchronous agent in a thread pool to avoid blocking
                    loop = asyncio.get_event_loop()
                    update_data = await loop.run_in_executor(
                        None, agent_callable, self.pipeline_state, **kwargs
                    )

                # Update the pipeline state with the results from the agent
                if update_data and isinstance(update_data, dict):
                    await self.pipeline_state.update_state(update_data)
                    logger.info(
                        f"Agent '{name}' executed successfully and updated state."
                    )
                elif update_data:
                    logger.warning(
                        f"Agent '{name}' executed but returned non-dict data. State not updated."
                    )
                else:
                    logger.info(
                        f"Agent '{name}' executed successfully. No state update returned."
                    )

            except Exception as e:
                logger.error(
                    f"Error executing agent '{name}': {e}", exc_info=True
                )
                # Optionally, emit a failure event
                # await self.pipeline_state.event_distributor.publish("agent_error", agent_name=name, error=str(e))

    def load_agents_from_config(self, agent_configs: Dict[str, Any]):
        """
        Loads and registers agents based on a configuration dictionary.
        This part would need a mechanism to import/resolve callables from strings.
        Example config:
        {
            "alpha_agent_1": {
                "module": "my_agents.alpha",
                "callable": "generate_signals",
                "trigger": "market_data_processed"
            },
            ...
        }
        """
        # This implementation is simplified. A real implementation would
        # need dynamic imports (importlib) to load callables.
        logger.warning(
            "load_agents_from_config is not fully implemented (requires dynamic import)."
        )
        # Example pseudo-code for dynamic loading:
        # for name, config in agent_configs.items():
        #     try:
        #         module = importlib.import_module(config['module'])
        #         callable_func = getattr(module, config['callable'])
        #         self.register_agent(name, callable_func, config['trigger'])
        #     except Exception as e:
        #         logger.error(f"Failed to load agent '{name}': {e}")
        pass

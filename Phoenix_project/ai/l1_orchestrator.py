# Phoenix_project/ai/l1_orchestrator.py
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from observability.metrics import L1_LAT
from schemas.fusion_result import L1AgentResult
from ai.agent_registry import L1_AGENTS
from pipeline_state import PipelineState
from typing import List

class L1Orchestrator:
    """
    Coordinates the execution of all registered L1 (Level 1) agents.
    L1 agents are specialized, independent AI agents that perform specific tasks
    like data retrieval, sentiment analysis, or fact-checking.
    """

    def run_l1_agents(self, state: PipelineState) -> List[L1AgentResult]:
        results = []

        def _run_and_time_agent(agent, agent_state):
            """Wrapper to run agent and measure its execution latency."""
            start_time = time.time()
            try:
                result_data = agent.run(agent_state)
                return agent, result_data, time.time() - start_time, None
            except Exception as e:
                return agent, None, time.time() - start_time, e

        with ThreadPoolExecutor(max_workers=len(L1_AGENTS)) as executor:
            future_to_task = {executor.submit(_run_and_time_agent, agent, state): agent.name for agent in L1_AGENTS}

            for future in as_completed(future_to_task):
                try:
                    agent, result_data, latency, error = future.result()
                    L1_LAT.observe(latency)
                    if error:
                        raise error

                    results.append(L1AgentResult(
                        agent_name=agent.name,
                        ticker=state.ticker,
                        output=result_data
                    ))
                except Exception as e:
                    agent_name = future_to_task[future]
                    print(f"Agent {agent_name} failed: {e}")
                    # TODO: Log this failure

        return results

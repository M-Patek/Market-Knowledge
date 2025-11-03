"""
L2 Agent: Planner
Refactored from reasoning/planner.py.
Responsible for L1 Task Initialization as per the blueprint.
"""
from typing import Any, Dict, List

from agents.l2.base import BaseL2Agent
from core.pipeline_state import PipelineState
from core.schemas.task_schema import TaskGraph

class PlannerAgent(BaseL2Agent):
    """
    Implements the L2 Planner agent.
    Inherits from BaseL2Agent and implements the run method
    to decompose the main task.
    """
    
    def run(self, state: PipelineState, evidence_items: List[Any] = None) -> TaskGraph:
        """
        Analyzes the main task from the pipeline state and generates a
        multi-step execution graph (subgoals and dependencies).
        
        Args:
            state (PipelineState): The current pipeline state, containing the main task.
            evidence_items (List[Any], optional): Not used by the planner. Defaults to None.
            
        Returns:
            TaskGraph: A Pydantic model defining the subgoals and dependencies.
        """
        # TODO: Implement actual planning logic (e.g., using Gemini call).
        # This is a mock plan based on the original reasoning/planner.py logic.
        
        # We assume the main task is accessible via the state.
        # This 'get_main_task_query' is a placeholder for state.main_task or similar.
        task_query_data = state.get_main_task_query() 
        
        # Mock logic from original file
        ticker = task_query_data.get("ticker", "UNKNOWN")
        
        graph_dict = {
            "subgoals": [f"analyze fundamentals for {ticker}", f"analyze technicals for {ticker}", f"run adversary on {ticker}"],
            "dependencies": {
                "fusion": ["analyze fundamentals", "analyze technicals", "run adversary"]
            }
        }
        
        return TaskGraph(**graph_dict)

    def __repr__(self) -> str:
        return f"<PlannerAgent(id='{self.agent_id}')>"

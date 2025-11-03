"""
Defines the Pydantic schema for task decomposition.
"""
from pydantic import BaseModel, Field
from typing import List, Dict

class TaskGraph(BaseModel):
    """
    Defines the execution graph for L1 agents, as determined by the L2 Planner.
    This is the output of the PlannerAgent.
    """
    subgoals: List[str] = Field(..., description="List of sub-task descriptions or IDs to execute.")
    dependencies: Dict[str, List[str]] = Field(..., description="Dependency graph, e.g., {'task_C': ['task_A', 'task_B']}.")

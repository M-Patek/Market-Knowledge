# Phoenix_project/reasoning/planner.py


def build_graph(task: dict) -> dict:
    """
    Analyzes the main task and generates a multi-step execution graph
    (subgoals and dependencies).
    
    Outputs sub-tasks and dependency relationships.
    """
    # TODO: 实现实际的规划逻辑 (例如，使用 Gemini 调用)。
    # 这是一个基于 Task 5 规范的模拟计划。
    
    ticker = task.get("ticker", "UNKNOWN")
    return {
        "subgoals": [f"analyze fundamentals for {ticker}", f"analyze technicals for {ticker}", f"run adversary on {ticker}"],
        "dependencies": {
            "fusion": ["analyze fundamentals", "analyze technicals", "run adversary"]
        }
    }

# Phoenix_project/fusion/uncertainty_guard.py
from typing import Dict, Any


def guard(result: dict, threshold: float = 0.4) -> dict:
    """
    If uncertainty > threshold -> re-reasoning or refusal to answer.
    Triggers re-reasoning or outputs a clarification statement.
    """
    
    # 我们从 fusion/synthesizer (Task 14) 获取不确定性
    uncertainty = result.get("uncertainty_score", 0.0)
    
    if uncertainty > threshold:
        result["status"] = "RE_REASONING_TRIGGERED"
        result["clarification"] = f"Analysis uncertainty ({uncertainty:.2f}) exceeds threshold ({threshold}). Triggering re-reasoning."
    else:
        result["status"] = "SUCCESS"
    
    return result

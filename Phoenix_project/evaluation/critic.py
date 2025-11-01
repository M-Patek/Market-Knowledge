# Phoenix_project/evaluation/critic.py
from typing import List, Dict, Any

def review(paths: list[dict]) -> dict:
    """
    Invokes Gemini for logic/fact review.
    Outputs a list of issues and suggestions for improvement.
    (Based on old ContradictionDetector)
    """
    
    # 这是一个复杂的矛盾检测模型 (Task 10) 的占位符。
    # 真正的实现会：
    # 1. 将 'paths' (agent 结果) 格式化为 Critic agent 的提示。
    # 2. 调用 Gemini (根据规范) 来审查逻辑矛盾或事实错误。
    # 3. 将 Gemini 输出解析为结构化的问题列表。
    
    # 模拟逻辑：如果一个路径说 "Buy" 而另一个说 "Hold"，则记录下来。
    sentiments = [p.get("result") for p in paths if p.get("result")]
    if "Buy" in sentiments and "Hold" in sentiments:
        issues = [{"type": "Contradiction", "details": "Agents disagree on Buy vs. Hold.", "suggestion": "Run arbitrator."}]
    else:
        issues = []
        
    return {"issues": issues, "suggestions": []}

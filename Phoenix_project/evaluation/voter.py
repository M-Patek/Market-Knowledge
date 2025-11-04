# Phoenix_project/evaluation/voter.py
from typing import List
from Phoenix_project.monitor.metrics import UNCERTAINTY # 来自 Task 25 的更正后导入

def vote(paths: list[dict]) -> dict:
    """
    Calculates consistency and the winning path(s).
    Uses a Bayesian approach to update beliefs and calculate a final,
    probabilistic sentiment.
    """
    
    if not paths:
        print("Warning: Voter received no paths.")
        return {"consistency_score": 0.0, "winner_ids": [], "final_sentiment": "Neutral", "confidence": 0.0}

    # --- 模拟贝叶斯融合逻辑 ---
    # 这是一个简化的占位符。真正的实现会：
    # 1. 定义先验信念 (例如 P(Bullish), P(Bearish), P(Neutral))。
    # 2. 为每个 agent 的输出建立似然模型 P(Evidence | Sentiment)。
    # 3. 应用贝叶斯定理计算后验概率 P(Sentiment | Evidence)。
    
    # 模拟逻辑：平均分数，计算投票
    sentiments = []
    confidences = []
    winner_ids = []
    
    for res in paths:
        # 我们假设路径结构来自 Task 4: {"agent":"...", "result":"...", "confidence":...}
        sentiments.append(res.get("result", "Neutral"))
        confidences.append(res.get("confidence", 0.0))
        winner_ids.append(res.get("agent")) # 暂定：目前所有路径都是 "winners"
    
    # 模拟一致性：( 'Buy' 结果的数量) / (总结果数)
    buy_count = sentiments.count("Buy")
    consistency_score = (buy_count / len(sentiments)) if sentiments else 0.0
    
    # 模拟获胜者
    if consistency_score > 0.6:
        final_sentiment = "Buy"
    elif consistency_score < 0.4:
        final_sentiment = "Hold" # 假设 "Hold" 是另一个主要情绪
    else:
        final_sentiment = "Neutral"
    
    # 模拟置信度
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    uncertainty_score = 1.0 - avg_confidence

    # 为 Prometheus 检测不确定性分数
    UNCERTAINTY.observe(uncertainty_score)
    
    return {
        "consistency_score": consistency_score,
        "winner_ids": winner_ids,
        "final_sentiment": final_sentiment,
        "confidence": avg_confidence,
        "uncertainty": uncertainty_score
    }

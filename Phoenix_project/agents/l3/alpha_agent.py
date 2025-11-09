# agents/l3/alpha_agent.py
import numpy as np
from typing import Optional

from .base import BaseDRLAgent
from Phoenix_project.core.schemas.fusion_result import FusionResult

class AlphaAgent(BaseDRLAgent):
    """
    L3 Alpha 智能体。
    负责根据 L2 分析和市场状态，决定理想的 *目标仓位* (例如 目标权重)。
    """

    def _format_obs(self, state_data: dict, fusion_result: Optional[FusionResult]) -> np.ndarray:
        """
        [任务 2.1] 格式化观察值以匹配 TradingEnv 的新 (5-d) 状态空间。
        
        Args:
            state_data (dict): 包含 {'balance', 'holdings', 'price'} 的实时数据。
            fusion_result (FusionResult): 来自 L2 认知引擎的分析结果。

        Returns:
            np.ndarray: 匹配 TradingEnv.observation_space 的 5-d 状态向量。
        """
        # 1. 从 state_data 中提取市场状态
        balance = state_data.get('balance', 0.0)
        holdings = state_data.get('holdings', 0.0)
        price = state_data.get('price', 0.0)

        # 2. (关键) 从 L2 FusionResult 中提取 L2 特征
        if fusion_result and hasattr(fusion_result, 'sentiment_score'):
            # [任务 1.1] 匹配 trading_env.py
            # 假设 fusion_result 有一个数值情感得分 (例如 -1.0 到 1.0)
            # (如果您的 FusionResult 使用 'final_decision' (BUY/SELL), 
            #  您需要在这里将其转换为数值，例如 BUY=1.0, SELL=-1.0, HOLD=0.0)
            sentiment = fusion_result.sentiment_score 
            confidence = fusion_result.confidence
        else:
            # 如果没有 L2 结果 (例如周期开始时)，提供默认值
            sentiment = 0.0  # 中性情感
            confidence = 0.5 # 中性信心

        # 3. 构建与 TradingEnv._get_state() 完全匹配的状态向量
        # 状态 (5-d): [balance, shares_held, price, l2_sentiment, l2_confidence]
        obs = np.array([
            balance,
            holdings,
            price,
            sentiment,
            confidence
        ], dtype=np.float32)
        
        return obs

# ---
# [主人喵的重要提示 🐱]
# 
# 主人喵！您需要对以下文件应用 *完全相同* 的 _format_obs 方法：
# 1. Phoenix_project/agents/l3/risk_agent.py
# 2. Phoenix_project/agents/l3/execution_agent.py
# 
# 确保所有 L3 智能体都使用这个新的 5-d 观察空间！
# ---

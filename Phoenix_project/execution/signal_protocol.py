"""
Execution Signal Protocol
Defines the data structure for signals sent from strategies to the execution layer.
"""
from pydantic import BaseModel, Field
from typing import Dict, Optional
import datetime

class StrategySignal(BaseModel):
    """
    关键修正 (Error 2 & 11): 
    将信号协议从 "单标的" 更改为 "完整投资组合目标"。
    OrderManager 期望收到一个包含所有目标权重的字典。
    """
    
    strategy_id: str
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.now)

    # 修正: 使用 "target_weights" (复数) 字典
    # 替代原有的 'ticker', 'action', 'target_weight'
    target_weights: Dict[str, float] = Field(
        ...,
        description="完整的投资组合目标权重, e.g., {'AAPL': 0.5, 'GOOG': 0.3, 'CASH': 0.2}"
    )

    # (可选) 增加一个元数据字段，说明信号的来源或类型
    metadata: Optional[Dict] = Field(default_factory=dict)

    class Config:
        validate_assignment = True

# 原始的(不正确的)定义已被移除:
# class StrategySignal(BaseModel):
#     ticker: str
#     action: str  # e.g., "BUY", "SELL", "HOLD"
#     target_weight: float # Desired portfolio weight
#     metadata: Optional[Dict] = None

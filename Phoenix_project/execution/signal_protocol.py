"""
信号协议 (Signal Protocol)
(这个文件在修复后变得很简单)

负责定义信号的结构。
"""

# FIX (E2): 从 data_schema 导入 Signal
# 修正：将 'core.schemas...' 转换为 'Phoenix_project.core.schemas...'
from Phoenix_project.core.schemas.data_schema import Signal

# FIX (E6): 移除了 StrategySignal 的导入，因为它不存在
# from .interfaces import StrategySignal 

class SignalProcessor:
    """
    (占位符) 
    在更复杂的系统中，这可能负责转换、验证或
    充实来自不同策略的信号。
    """
    
    def __init__(self):
        pass

    def validate_signal(self, signal: Signal) -> bool:
        """
        验证信号是否有效。
        """
        if signal.strength < 0 or signal.strength > 1:
            return False
        if signal.signal_type not in ("BUY", "SELL", "HOLD"):
            return False
        return True

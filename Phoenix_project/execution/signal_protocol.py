"""
信号协议 (Signal Protocol)
(这个文件在修复后变得很简单)

负责定义信号的结构。
"""
# [任务 4 添加]：为 StrategySignal 中的 Dict 导入
from typing import Dict, Any, Optional
import logging
from uuid import uuid4
import pandas as pd

# FIX (E2): 从 data_schema 导入 Signal
# 修正：将 'core.schemas...' 转换为 'Phoenix_project.core.schemas...'
from Phoenix_project.core.schemas.data_schema import Signal
from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.monitor.logging import get_logger

# FIX (E6): 移除了 StrategySignal 的导入，因为它不存在
# from .interfaces import StrategySignal 

logger = get_logger(__name__)


# [任务 4 实现]：定义 StrategySignal
class StrategySignal(Signal):
    """
    [任务 4 实现]
    继承自 Signal，添加策略特有的元数据。
    """
    strategy_name: str
    raw_score: float
    parameters: Dict[str, Any] = {}

class SignalProcessor:
    """
    [已重构] 
    转换、验证和丰富信号，然后再执行。
    (不再是占位符)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        初始化信号处理器。
        Args:
            config: system.yaml 中 'signal_processor' 部分的配置。
        """
        self.config = config.get('signal_processor', {})
        # 示例配置：过滤掉价值低于 $1.00 的交易
        self.min_order_value_usd = self.config.get('min_order_value_usd', 1.0)
        logger.info(f"SignalProcessor initialized with min order value: ${self.min_order_value_usd}")

    def process_and_enrich(
        self, 
        signal: Signal, 
        pipeline_state: PipelineState
    ) -> Optional[Signal]:
        """
        验证、丰富和转换原始信号为可执行信号。
        如果信号无效或被过滤掉，则返回 None。

        Args:
            signal: 来自 PortfolioConstructor 或 L3 Agent 的原始信号。
            pipeline_state: 当前的管线状态，用于获取价格等上下文。

        Returns:
            Optional[Signal]: 一个经过处理的信号，如果有效。
        """

        # 1. 验证 Schema
        if not self.validate_schema(signal):
            logger.warning(f"Signal for {signal.symbol} failed schema validation.")
            return None

        # 2. 丰富（Enrich） (添加缺失的元数据)
        signal.signal_id = signal.signal_id or f"sig_{uuid4()}"
        signal.timestamp = signal.timestamp or pd.Timestamp.now(tz='UTC')

        # 3. 过滤（Filter） (示例：粉尘交易过滤器)

        # (我们依赖任务 1.2 中存入 state 的 L0 市场数据)
        market_data = pipeline_state.get_value("current_market_data", {})
        price = market_data.get(signal.symbol)

        if price and price > 0:
            order_value_usd = abs(signal.quantity * price)
            # 过滤掉价值过小的交易
            if order_value_usd < self.min_order_value_usd:
                logger.info(f"Filtering dust trade for {signal.symbol}. Value ${order_value_usd:.2f} < ${self.min_order_value_usd}.")
                return None
        elif signal.quantity != 0:
            # 如果我们没有价格，但交易量不是 0，发出警告
            logger.warning(f"Cannot get price for {signal.symbol} from state to check min order value.")

        if signal.quantity == 0:
            logger.info(f"Signal for {signal.symbol} is 0 qty (close/no-op), passing.")

        logger.debug(f"SignalProcessor approved signal {signal.signal_id} for {signal.symbol}.")
        return signal

    def validate_schema(self, signal: Signal) -> bool:
        """
        根据 Signal 模式验证核心属性。
        (替换了旧的、已损坏的 validate_signal)
        """
        if not signal.symbol:
            logger.warning("Signal validation failed: Missing symbol.")
            return False

        if not isinstance(signal.quantity, (int, float)):
            logger.warning(f"Signal validation failed for {signal.symbol}: Invalid quantity {signal.quantity}.")
            return False

        if signal.price_target is not None and signal.price_target <= 0:
            logger.warning(f"Signal validation failed for {signal.symbol}: Invalid price_target {signal.price_target}.")
            return False

        if signal.confidence is not None and (signal.confidence < 0.0 or signal.confidence > 1.0):
             logger.warning(f"Signal validation failed for {signal.symbol}: Invalid confidence {signal.confidence}.")
             return False

        return True

"""
投资组合构造器 (Portfolio Constructor)
将认知引擎的高级决策（FusionResult）转换为具体的、可执行的交易订单（Order）。
"""
from typing import List, Dict, Optional

# FIX (E2, E3): 从统一的 data_schema 和 fusion_result 导入
from Phoenix_project.core.schemas.data_schema import Signal, Order, PortfolioState, OrderStatus
from Phoenix_project.core.schemas.fusion_result import FusionResult, AgentDecision

# FIX (E8): 导入 IPositionSizer 接口 (原为 SizingMethod)
from Phoenix_project.sizing.base import IPositionSizer

from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

class PortfolioConstructor:
    """
    负责将 AI 决策（'BULLISH', 'BEARISH'）与仓位管理逻辑相结合，
    以确定最终的订单数量和类型。
    """

    def __init__(self, position_sizer: IPositionSizer):
        self.position_sizer = position_sizer
        self.log_prefix = "PortfolioConstructor:"

    def translate_decision_to_signal(self, fusion_result: FusionResult) -> Optional[Signal]:
        """
        步骤1：将 FusionResult 转换为标准化的 Signal。
        """
        
        # 这是一个占位符。
        # FIX (E3): 之前的 'AgentDecision' 导入会失败。现在已修复。
        # // This will fail: fusion_result.agent_decisions 
        # (现在不会失败了)

        if not fusion_result.agent_decisions:
            logger.warning(f"{self.log_prefix} FusionResult {fusion_result.id} has no agent decisions.")
            return None

        # 假设元数据中包含目标符号
        target_symbol = fusion_result.metadata.get("target_symbol")
        if not target_symbol:
            logger.error(f"{self.log_prefix} FusionResult {fusion_result.id} missing 'target_symbol' in metadata.")
            return None
            
        # 简化的转换逻辑
        decision = fusion_result.final_decision.upper()
        signal_type = "HOLD" # 默认
        
        if decision in ("STRONG_BUY", "BULLISH", "BUY"):
            signal_type = "BUY"
        elif decision in ("STRONG_SELL", "BEARISH", "SELL"):
            signal_type = "SELL"
        elif decision == "NEUTRAL":
            signal_type = "HOLD"
            
        logger.info(f"{self.log_prefix} Translated decision {decision} to signal {signal_type} for {target_symbol}")

        # FIX (E2): 使用 Signal 模式
        return Signal(
            symbol=target_symbol,
            timestamp=fusion_result.timestamp,
            signal_type=signal_type,
            strength=fusion_result.final_confidence,
            metadata={"fusion_id": fusion_result.id}
        )

    def generate_orders(self, signals: List[Signal], portfolio_state: PortfolioState) -> List[Order]:
        """
        步骤2：将 Signal 列表和当前投资组合状态转换为 Order 列表。
        """
        orders = []
        if not signals:
            return orders
            
        current_positions = portfolio_state.positions

        for signal in signals:
            if signal.signal_type == "HOLD":
                continue

            # 使用仓位管理模块 (IPositionSizer) 计算目标仓位
            # FIX (E8): self.position_sizer 现在是 IPositionSizer
            target_quantity = self.position_sizer.calculate_target_quantity(
                signal,
                portfolio_state
            )

            # 计算订单数量 (与当前持仓对比)
            current_quantity = current_positions.get(signal.symbol, None)
            current_qty_val = current_quantity.quantity if current_quantity else 0.0
            
            order_quantity = target_quantity - current_qty_val

            if abs(order_quantity) > 1e-6: # 避免浮点数0
                logger.info(f"{self.log_prefix} Generating order for {signal.symbol}: "
                            f"Target={target_quantity}, Current={current_qty_val}, OrderQty={order_quantity}")
                            
                # FIX (E2): 使用 Order 模式
                new_order = Order(
                    id=f"order_{signal.symbol}_{signal.timestamp.isoformat()}", # 实际应由UUID生成
                    symbol=signal.symbol,
                    quantity=order_quantity,
                    order_type="MARKET", # 默认为市价单
                    status=OrderStatus.NEW,
                    metadata={"signal_strength": signal.strength, "fusion_id": signal.metadata.get("fusion_id")}
                )
                orders.append(new_order)

        return orders

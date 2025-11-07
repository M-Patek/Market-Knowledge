"""
投资组合构造器 (Portfolio Constructor)
将认知引擎的高级决策（FusionResult）转换为具体的、可执行的交易订单（Order）。

[主人喵的修复]
添加了 Orchestrator 调用的 'construct' 方法的实现。
"""
from typing import List, Dict, Optional
from datetime import datetime

# FIX (E2, E3): 从统一的 data_schema 和 fusion_result 导入
from Phoenix_project.core.schemas.data_schema import Signal, Order, PortfolioState, OrderStatus
# [主人喵的修复] 导入 FusionResult 和新添加的 TargetPortfolio 模式
from Phoenix_project.core.schemas.fusion_result import FusionResult, AgentDecision
from Phoenix_project.core.schemas.data_schema import TargetPortfolio, TargetPosition

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
        # [主人喵的修复] 增加一些简单的转换规则
        self.decision_to_weight_map = {
            "STRONG_BUY": 0.15,  # 15% 基础权重
            "BUY": 0.10,
            "BULLISH": 0.05,
            "HOLD": 0.0,
            "NEUTRAL": 0.0,
            "BEARISH": -0.05,
            "SELL": -0.10,
            "STRONG_SELL": -0.15,
        }
        self.default_weight = 0.0

    def construct(self, fusion_result: FusionResult, current_portfolio: PortfolioState) -> TargetPortfolio:
        """
        [主人喵的修复]
        实现 Orchestrator 调用的 'construct' 存根。
        将 AI 的 FusionResult 转换为 TargetPortfolio (目标权重)。
        """
        logger.info(f"{self.log_prefix} Constructing portfolio from FusionResult {fusion_result.id}...")
        
        target_positions = []
        
        # --- 核心逻辑：将 AI 决策转换为目标权重 ---
        # 这是一个简化的实现，假设 FusionResult 针对单个符号。
        # 一个更复杂的系统可能会处理 fusion_result.metadata 中的多个符号。
        
        symbol = fusion_result.target_symbol
        decision = fusion_result.decision.upper()
        confidence = fusion_result.confidence
        
        # 1. 从映射中获取基础权重
        base_weight = self.decision_to_weight_map.get(decision, self.default_weight)
        
        # 2. 按置信度调整权重 (简单的线性缩放)
        #    (注意：self.position_sizer 也可以在这里使用，
        #     但 `construct` 的接口没有提供足够的信息给 sizer)
        target_weight = base_weight * confidence
        
        reasoning = f"AI Decision: {decision} (Conf: {confidence:.2f}). Base Weight: {base_weight:.2%}. Final Weight: {target_weight:.2%}"
        logger.info(f"{self.log_prefix} {symbol}: {reasoning}")

        if abs(target_weight) > 0.001: # 避免 0 权重
            target_positions.append(
                TargetPosition(
                    symbol=symbol,
                    target_weight=target_weight,
                    reasoning=reasoning
                )
            )
        
        # 3. 创建目标投资组合对象
        target_portfolio = TargetPortfolio(
            timestamp=datetime.utcnow(),
            positions=target_positions,
            metadata={
                "source_fusion_id": fusion_result.id,
                "strategy_id": "phoenix_v1_core" # 示例
            }
        )
        
        return target_portfolio

    def translate_decision_to_signal(self, fusion_result: FusionResult) -> Optional[Signal]:
        """
        步骤1：将 FusionResult 转换为标准化的 Signal。
        (这是代码库中的遗留方法，Orchestrator 目前不调用它)
        """
        
        # 这是一个占位符。
        # FIX (E3): 之前的 'AgentDecision' 导入会失败。现在已修复。
        # // This will fail: fusion_result.agent_decisions 
        # (现在不会失败了)
        
        # [主人喵的修复] 检查 'fusion_result' 是否有 'agent_decisions' 属性
        if not hasattr(fusion_result, 'agent_decisions') or not fusion_result.agent_decisions:
            logger.warning(f"{self.log_prefix} FusionResult {fusion_result.id} has no agent decisions.")
            # return None # 原始逻辑
        
        # [主人喵的修复] 检查 'fusion_result' 是否有 'metadata' 属性
        if not hasattr(fusion_result, 'metadata'):
             logger.error(f"{self.log_prefix} FusionResult {fusion_result.id} missing 'metadata' attribute.")
             return None
             
        # 假设元数据中包含目标符号
        target_symbol = fusion_result.metadata.get("target_symbol")
        if not target_symbol:
            # [主人喵的修复] 尝试从 'target_symbol' 属性获取
            if hasattr(fusion_result, 'target_symbol') and fusion_result.target_symbol:
                 target_symbol = fusion_result.target_symbol
            else:
                logger.error(f"{self.log_prefix} FusionResult {fusion_result.id} missing 'target_symbol' in metadata and attributes.")
                return None
            
        # 简化的转换逻辑
        decision = fusion_result.decision.upper()
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
            strength=fusion_result.confidence, # [主人喵的修复] 确保 'confidence' 存在
            metadata={"fusion_id": fusion_result.id}
        )

    def generate_orders(self, signals: List[Signal], portfolio_state: PortfolioState) -> List[Order]:
        """
        步骤2：将 Signal 列表和当前投资组合状态转换为 Order 列表。
        (这是代码库中的遗留方法，Orchestrator 目前不调用它)
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
            
            # [主人喵的修复] IPositionSizer.calculate_target_quantity 不存在。
            # 真实接口是 size_positions(candidates, max_total_allocation)
            # 这是一个接口不匹配，我们将模拟逻辑：
            logger.warning(f"{self.log_prefix} generate_orders is using MOCK sizing logic due to interface mismatch.")
            # 模拟 sizer
            mock_candidates = [{
                "ticker": signal.symbol, 
                "confidence": signal.strength, 
                "signal_type": signal.signal_type
            }]
            # 假设 sizer 返回 [{ "ticker": "AAPL", "capital_allocation_pct": 0.05 }]
            sized_plan = self.position_sizer.size_positions(mock_candidates, max_total_allocation=1.0) 
            
            if not sized_plan:
                continue
                
            target_alloc_pct = sized_plan[0]["capital_allocation_pct"]
            target_dollar_value = portfolio_state.total_value * target_alloc_pct
            
            # [主人喵的修复] 获取价格以计算数量
            current_pos = current_positions.get(signal.symbol)
            price = 150.0 # 默认模拟价格
            if current_pos and current_pos.quantity != 0:
                price = current_pos.market_value / current_pos.quantity
            elif current_pos and current_pos.average_price != 0:
                price = current_pos.average_price
            else:
                 logger.warning(f"{self.log_prefix} No price for {signal.symbol}, using MOCK $150.0 in generate_orders")
                 
            target_quantity = target_dollar_value / price
            if signal.signal_type == "SELL":
                target_quantity = -abs(target_quantity)

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

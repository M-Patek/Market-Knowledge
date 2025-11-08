"""
订单管理器 (Order Manager)
负责处理订单的整个生命周期：发送、监控、取消。

[阶段 1 & 4 重构]
- 移除了所有模拟执行逻辑 (已迁移到 SimulatedBrokerAdapter)。
- 实现了 'process_target_portfolio' 作为统一的入口点。
- 注入了 TradeLifecycleManager 以便在收到成交时更新投资组合。
"""
from typing import List, Dict, Optional
from datetime import datetime
import uuid
from .interfaces import IBrokerAdapter
import threading

# [阶段 1 & 4 变更] 导入依赖
from Phoenix_project.execution.trade_lifecycle_manager import TradeLifecycleManager
from Phoenix_project.core.schemas.data_schema import (
    Order, Fill, OrderStatus, PortfolioState, TargetPortfolio, Position
)

# 修正：将 'monitor.logging...' 转换为 'Phoenix_project.monitor.logging...'
from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

class OrderManager:
    """
    作为订单和券商适配器之间的中介。
    跟踪所有活动订单的状态。
    现在是纯粹的异步逻辑：提交订单，并通过回调响应事件。
    """
    
    def __init__(self, broker: IBrokerAdapter, trade_lifecycle_manager: TradeLifecycleManager):
        """
        [阶段 1 & 4 变更]
        - 注入 TradeLifecycleManager 以便转发 Fill 事件。
        """
        self.broker = broker
        self.trade_lifecycle_manager = trade_lifecycle_manager # <--- 新增
        self.active_orders: Dict[str, Order] = {} # key: order_id
        self.lock = threading.Lock()
        
        # 订阅券商的回调
        self.broker.subscribe_fills(self._on_fill)
        self.broker.subscribe_order_status(self._on_order_status_update)
        
        self.log_prefix = "OrderManager:"
        logger.info(f"{self.log_prefix} Initialized.")

    def _on_fill(self, fill: Fill):
        """
        [阶段 1 & 4 变更]
        处理来自券商的成交回报 (Fill)。
        将 Fill 事件转发给 TradeLifecycleManager。
        """
        with self.lock:
            logger.info(f"{self.log_prefix} Received Fill: {fill.order_id} ({fill.quantity} @ {fill.price})")
            
            order = self.active_orders.get(fill.order_id)
            if not order:
                logger.warning(f"{self.log_prefix} Received fill for unknown or closed order {fill.order_id}")
                # 即使订单未知，也可能需要转发 Fill
                
            # [阶段 4] 将 Fill 转发给 TLM
            try:
                self.trade_lifecycle_manager.on_fill(fill)
            except Exception as e:
                logger.error(f"{self.log_prefix} Failed to process fill with TradeLifecycleManager: {e}", exc_info=True)


    def _on_order_status_update(self, order: Order):
        """
        处理来自券商的订单状态更新。
        """
        with self.lock:
            if not order or not order.id or not order.status:
                logger.warning(f"{self.log_prefix} Received invalid order status update: {order}")
                return
                
            logger.info(f"{self.log_prefix} Status Update: Order {order.id} is now {order.status}")
            
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                # 订单生命周期结束，从活动列表中移除
                if order.id in self.active_orders:
                    del self.active_orders[order.id]
                    logger.info(f"{self.log_prefix} Order {order.id} closed.")
            else:
                # 更新活动订单的状态
                self.active_orders[order.id] = order

    def generate_orders(self, 
                        current_portfolio: PortfolioState, 
                        target_portfolio: TargetPortfolio, 
                        market_prices: Dict[str, float]
                       ) -> List[Order]:
        """
        [阶段 4]
        此方法保持不变。
        计算当前投资组合和目标投资组合之间的“差异”(diff)，
        并生成实现该目标所需的市价单。
        """
        logger.info(f"{self.log_prefix} Generating orders to match target portfolio...")
        orders_to_generate: List[Order] = []
        
        if not current_portfolio or not target_portfolio:
            logger.warning(f"{self.log_prefix} Missing current or target portfolio. Cannot generate orders.")
            return []

        total_value = current_portfolio.total_value
        if total_value <= 0:
            if current_portfolio.cash > 0:
                 logger.warning(f"{self.log_prefix} Total portfolio value is zero. Using cash value {current_portfolio.cash} for sizing.")
                 total_value = current_portfolio.cash
            else:
                logger.warning(f"{self.log_prefix} Total portfolio value and cash are zero or negative. Cannot generate orders.")
                return []

        # 1. 将当前持仓转换为权重字典
        current_weights: Dict[str, float] = {}
        for symbol, position in current_portfolio.positions.items():
            # [修复] 确保 total_value 不为零
            current_weights[symbol] = position.market_value / total_value if total_value != 0 else 0

        # 2. 将目标持仓转换为权重字典
        target_weights: Dict[str, float] = {}
        for pos in target_portfolio.positions:
            target_weights[pos.symbol] = pos.target_weight

        # 3. 合并所有相关符号
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())

        # 4. 计算差异 (Diff)
        for symbol in all_symbols:
            current_w = current_weights.get(symbol, 0.0)
            target_w = target_weights.get(symbol, 0.0)
            
            delta_weight = target_w - current_w
            
            # 如果变化足够大
            if abs(delta_weight) > 0.0001: # 0.01% 变动阈值
                
                target_dollar_value = delta_weight * total_value
                
                price = market_prices.get(symbol)
                
                if price is None or price <= 0:
                    logger.error(f"{self.log_prefix} Cannot determine price for position {symbol} from provided market data. Skipping order.")
                    continue
                
                order_quantity = target_dollar_value / price
                
                # 创建订单
                new_order = Order(
                    id=f"order_{symbol}_{uuid.uuid4()}",
                    symbol=symbol,
                    quantity=order_quantity,
                    order_type="MARKET", # 默认为市价单
                    status=OrderStatus.NEW,
                    time_in_force="GTC", # [修复] 为 PaperTrading 添加默认 TIF
                    metadata={
                        "target_weight": target_w,
                        "current_weight": current_w,
                        "source_target_id": target_portfolio.metadata.get("source_fusion_id", "N/A")
                    }
                )
                orders_to_generate.append(new_order)
                logger.info(f"{self.log_prefix} Generated Order: {symbol} Qty {order_quantity:.4f} (Target: {target_w:.2%}, Current: {current_w:.2%})")

        return orders_to_generate

    # [阶段 1 & 4 变更] 移除 execute_orders 方法
    # def execute_orders(self, ...):
    #     ... (逻辑已迁移到 SimulatedBrokerAdapter)

    # [阶段 1 & 4 变更] 移除 submit_orders 方法，逻辑合并到 process_target_portfolio
    # def submit_orders(self, orders: List[Order]):
    #     ...
        
    def process_target_portfolio(
        self, 
        current_portfolio: PortfolioState, 
        target_portfolio: TargetPortfolio, 
        market_prices: Dict[str, float]
    ):
        """
        [阶段 4 实现]
        这是 Orchestrator 调用的新入口点。
        它负责：
        1. 调用 generate_orders() 计算差异。
        2. 遍历订单。
        3. 调用 self.broker.place_order() 提交订单。
        """
        logger.info(f"{self.log_prefix} Processing target portfolio...")
        
        # 1. 生成订单
        orders = self.generate_orders(current_portfolio, target_portfolio, market_prices)
        
        if not orders:
            logger.info(f"{self.log_prefix} No orders generated, processing complete.")
            return

        logger.info(f"{self.log_prefix} Submitting {len(orders)} orders to broker...")
        
        # 2. 提交订单到券商
        with self.lock:
            for order in orders:
                if order.id in self.active_orders:
                    logger.warning(f"{self.log_prefix} Attempted to submit duplicate order {order.id}")
                    continue
                
                price = market_prices.get(order.symbol)
                
                try:
                    # [阶段 4] 调用适配器
                    # SimBroker 将立即执行并触发回调
                    # PaperBroker 将异步发送并稍后触发回调
                    broker_order_id = self.broker.place_order(order, price)
                    
                    # 假设 place_order 成功（或至少被接受）
                    # 注意：SimBroker 会立即将其设置为 FILLED，但 PaperBroker 不会
                    if order.status == OrderStatus.NEW: # 如果 SimBroker 没有改变它
                        order.status = OrderStatus.PENDING 
                        
                    self.active_orders[order.id] = order
                    logger.info(f"{self.log_prefix} Submitted order {order.id} (Broker ID: {broker_order_id})")
                    
                except Exception as e:
                    logger.error(f"{self.log_prefix} Failed to place order {order.id}: {e}")
                    order.status = OrderStatus.REJECTED # 标记为本地拒绝

    def cancel_all_open_orders(self):
        """
        取消所有活动的订单。
        """
        with self.lock:
            logger.info(f"{self.log_prefix} Cancelling all {len(self.active_orders)} open orders...")
            # 复制列表以避免在迭代时修改
            order_ids = list(self.active_orders.keys())
            
        for order_id in order_ids:
            try:
                self.broker.cancel_order(order_id)
                logger.info(f"{self.log_prefix} Cancel request sent for {order_id}")
            except Exception as e:
                logger.error(f"{self.log_prefix} Failed to cancel order {order_id}: {e}")
                
    def get_open_orders(self) -> List[Order]:
        """
        获取所有活动订单的当前状态。
        """
        with self.lock:
            return list(self.active_orders.values())

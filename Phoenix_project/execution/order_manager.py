"""
订单管理器 (Order Manager)
负责处理订单的整个生命周期：发送、监控、取消。

[主人喵的修复]
添加了 Orchestrator 调用的 'generate_orders' 和 'execute_orders' 方法的实现。
"""
from typing import List, Dict, Optional
from datetime import datetime
import uuid
from .interfaces import IBrokerAdapter
import threading

# FIX (E2, E4): 从核心模式导入 Order, Fill, OrderStatus
# [主人喵的修复] 并导入 PortfolioState, TargetPortfolio
from Phoenix_project.core.schemas.data_schema import Order, Fill, OrderStatus, PortfolioState, TargetPortfolio, Position

# 修正：将 'monitor.logging...' 转换为 'Phoenix_project.monitor.logging...'
from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

class OrderManager:
    """
    作为订单和券商适配器之间的中介。
    跟踪所有活动订单的状态。
    
    [主人喵的修复]
    实现了 'generate_orders' (diff 逻辑) 和 'execute_orders' (模拟执行)。
    """
    
    def __init__(self, broker: IBrokerAdapter):
        self.broker = broker
        self.active_orders: Dict[str, Order] = {} # key: order_id
        self.lock = threading.Lock()
        
        # 订阅券商的回调
        self.broker.subscribe_fills(self._on_fill)
        self.broker.subscribe_order_status(self._on_order_status_update)
        
        self.log_prefix = "OrderManager:"
        logger.info(f"{self.log_prefix} Initialized.")

    def _on_fill(self, fill: Fill):
        """
        处理来自券商的成交回报 (Fill)。
        """
        with self.lock:
            logger.info(f"{self.log_prefix} Received Fill: {fill.order_id} ({fill.quantity} @ {fill.price})")
            
            order = self.active_orders.get(fill.order_id)
            if not order:
                logger.warning(f"{self.log_prefix} Received fill for unknown order {fill.order_id}")
                return
                
            # 这里的逻辑需要更复杂，以处理部分成交 (PARTIALLY_FILLED)
            # 为简化起见，我们假设一个 Fill 意味着订单
            # 状态将在 _on_order_status_update 中更新

            # TODO: 将 Fill 事件转发给 TradeLifecycleManager
            # (在 Orchestrator v2 中，TLM 是通过 'execute_orders' 的返回来更新的)

    def _on_order_status_update(self, order: Order):
        """
        处理来自券商的订单状态更新。
        """
        with self.lock:
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
                        market_prices: Dict[str, float]  # [主人喵的修复] 接收价格
                       ) -> List[Order]:
        """
        [主人喵的修复]
        实现 'generate_orders' 存根。
        
        核心逻辑：计算当前投资组合和目标投资组合之间的“差异”(diff)，
        并生成实现该目标所需的市价单。
        """
        logger.info(f"{self.log_prefix} Generating orders to match target portfolio...")
        orders_to_generate: List[Order] = []
        
        if not current_portfolio or not target_portfolio:
            logger.warning(f"{self.log_prefix} Missing current or target portfolio. Cannot generate orders.")
            return []

        total_value = current_portfolio.total_value
        if total_value <= 0:
            # [主人喵的修复] 允许在总价值为 0 时根据现金计算
            if current_portfolio.cash > 0:
                 logger.warning(f"{self.log_prefix} Total portfolio value is zero. Using cash value {current_portfolio.cash} for sizing.")
                 total_value = current_portfolio.cash
            else:
                logger.warning(f"{self.log_prefix} Total portfolio value and cash are zero or negative. Cannot generate orders.")
                return []

        # 1. 将当前持仓转换为权重字典
        current_weights: Dict[str, float] = {}
        for symbol, position in current_portfolio.positions.items():
            current_weights[symbol] = position.market_value / total_value

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
                
                # 5. [主人喵的修复] 从传入的 market_prices 获取可靠的当前价格
                price = market_prices.get(symbol)
                
                if price is None or price <= 0:
                    # 无法从 Orchestrator 获取价格
                    logger.error(f"{self.log_prefix} Cannot determine price for position {symbol} from provided market data. Skipping order.")
                    continue
                
                # [主人喵的修复] 移除旧的、不可靠的价格推断逻辑
                # current_pos = current_portfolio.positions.get(symbol) ...
                # if price <= 0: ...

                order_quantity = target_dollar_value / price
                
                # 创建订单
                new_order = Order(
                    id=f"order_{symbol}_{uuid.uuid4()}",
                    symbol=symbol,
                    quantity=order_quantity,
                    order_type="MARKET", # 默认为市价单
                    status=OrderStatus.NEW,
                    metadata={
                        "target_weight": target_w,
                        "current_weight": current_w,
                        "source_target_id": target_portfolio.metadata.get("source_fusion_id", "N/A")
                    }
                )
                orders_to_generate.append(new_order)
                logger.info(f"{self.log_prefix} Generated Order: {symbol} Qty {order_quantity:.4f} (Target: {target_w:.2%}, Current: {current_w:.2%})")

        return orders_to_generate

    def execute_orders(self, 
                       orders: List[Order], 
                       market_prices: Dict[str, float] # [主人喵的修复] 接收价格
                      ) -> List[Fill]:
        """
        [主人喵的修复]
        实现 'execute_orders' 存根。
        
        在回测环境中，这会立即模拟成交。
        在真实环境中，这将调用 self.submit_orders 并且什么也不返回
        (成交将通过回调异步到达)。
        
        Orchestrator 期望一个同步的 Fill 列表，所以我们模拟它。
        """
        fills = []
        logger.info(f"{self.log_prefix} Simulating execution (stub logic) for {len(orders)} orders...")
        
        for order in orders:
            if order.order_type != "MARKET":
                logger.warning(f"{self.log_prefix} Stub logic can only execute MARKET orders. Skipping {order.id}")
                continue
            
            # --- [主人喵的修复] 移除模拟价格，使用从 Orchestrator 传入的真实价格 ---
            fill_price = market_prices.get(order.symbol)
            
            if fill_price is None or fill_price <= 0:
                logger.warning(f"{self.log_prefix} Stub logic: No market price for {order.symbol}. Skipping fill for order {order.id}")
                continue
                
            # [主人喵的修复] 移除 MOCK_EXECUTION_PRICE 和 logger.critical 警告

            # 模拟滑点 (来自 SimulatedBrokerAdapter)
            slippage = 0.001 
            if order.quantity > 0: # Buy
                fill_price *= (1 + slippage)
            else: # Sell
                fill_price *= (1 - slippage)

            fill = Fill(
                id=f"fill-{order.id}",
                order_id=order.id,
                symbol=order.symbol,
                timestamp=datetime.utcnow(),
                quantity=order.quantity,
                price=fill_price,
                commission=0.0 # 模拟 0 佣金
            )
            fills.append(fill)
            
            # (重要) 立即将订单提交到我们自己的活动列表，
            # 并模拟券商的回调，以便 OrderManager 的状态保持一致
            order.status = OrderStatus.PENDING
            self.active_orders[order.id] = order
            self._on_fill(fill) # 模拟成交回调
            order.status = OrderStatus.FILLED
            self._on_order_status_update(order) # 模拟状态更新回调

        logger.info(f"{self.log_prefix} Stub execution complete. Generated {len(fills)} fills.")
        return fills

    def submit_orders(self, orders: List[Order]):
        """
        向券商提交一批新订单。
        (这是代码库中的原始方法，'execute_orders' 存根现在调用它)
        """
        if not orders:
            return
            
        with self.lock:
            for order in orders:
                if order.id in self.active_orders:
                    logger.warning(f"{self.log_prefix} Attempted to submit duplicate order {order.id}")
                    continue
                
                try:
                    broker_order_id = self.broker.place_order(order)
                    # broker_order_id 可能与 order.id 不同
                    # 真实系统中需要处理这种映射
                    
                    order.status = OrderStatus.PENDING # 假设 place_order 是异步的
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

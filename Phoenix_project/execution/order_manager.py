import asyncio
import logging
from typing import Dict, List, Optional, Set
from decimal import Decimal
from datetime import datetime

from Phoenix_project.core.schemas.data_schema import Order, OrderStatus, PortfolioState, TargetPortfolio, OrderSide, OrderType
from Phoenix_project.execution.interfaces import IBrokerAdapter
from Phoenix_project.data_manager import DataManager

logger = logging.getLogger(__name__)

class OrderManager:
    """
    订单管理器 (Order Manager)
    负责将 TargetPortfolio 转换为具体的 Order，并管理订单生命周期。
    [Task 5.2] Concurrent Execution & Zombie Position Fix
    """

    def __init__(self, broker_adapter: Optional[IBrokerAdapter], data_manager: DataManager):
        self.broker = broker_adapter
        self.data_manager = data_manager
        self.active_orders: Dict[str, Order] = {}
        self.log_prefix = "OrderManager:"
        self.lock = asyncio.Lock()

    async def reconcile_portfolio(self, state: PortfolioState):
        """
        根据当前状态和目标投资组合生成并提交订单。
        [Full Implementation] 包含完整的 Diff 计算逻辑和并发提交。
        """
        target_portfolio = state.target_portfolio
        if not target_portfolio: return

        # [Task 3.2] Fix: Fetch prices for both Target AND Current positions
        # Prevents "Zombie Position" crashes where we hold an asset not in target
        current_holdings = set(state.portfolio_state.positions.keys()) if state.portfolio_state else set()
        target_holdings = {p.symbol for p in target_portfolio.positions}
        relevant_symbols = current_holdings.union(target_holdings)

        market_prices: Dict[str, Decimal] = {}
        
        # 获取所有相关资产的最新价格
        for sym in relevant_symbols:
            md = await self.data_manager.get_latest_market_data(sym)
            if not md:
                logger.warning(f"{self.log_prefix} Market data missing for {sym}. Skipping.")
                continue
            market_prices[sym] = Decimal(str(md.close))

        orders_to_submit: List[Order] = []
        current_positions = state.portfolio_state.positions if state.portfolio_state else {}

        # --- 核心 Diff 逻辑 (此前被略写的部分) ---
        for symbol in relevant_symbols:
            if symbol not in market_prices:
                continue

            price = market_prices[symbol]
            if price <= 0: continue

            # 获取当前持仓数量
            current_qty = Decimal("0.0")
            if symbol in current_positions:
                current_qty = Decimal(str(current_positions[symbol].quantity))

            # 获取目标持仓数量
            target_qty = Decimal("0.0")
            # 查找目标组合中的匹配项
            target_pos = next((p for p in target_portfolio.positions if p.symbol == symbol), None)
            if target_pos:
                target_qty = Decimal(str(target_pos.quantity))

            # 计算差异
            delta = target_qty - current_qty
            
            # 设置最小交易阈值 (例如 0.0001) 防止极小订单
            if abs(delta) < Decimal("0.0001"):
                continue

            # 生成订单
            side = OrderSide.BUY if delta > 0 else OrderSide.SELL
            abs_qty = float(abs(delta))
            
            # 创建订单对象
            # 注意：实际生产中可能需要考虑 limit_price (限价) 或 slippage (滑点)
            order = Order(
                symbol=symbol,
                side=side,
                order_type=OrderType.MARKET, # 默认为市价单以保证成交
                quantity=abs_qty,
                timestamp=datetime.now(),
                status=OrderStatus.PENDING_SUBMISSION
            )
            
            # 如果是限价单逻辑 (可选):
            # order.limit_price = float(price * Decimal("1.01")) if side == OrderSide.BUY else float(price * Decimal("0.99"))

            orders_to_submit.append(order)
            
            # 将订单加入活跃列表跟踪
            async with self.lock:
                self.active_orders[order.id] = order

        logger.info(f"{self.log_prefix} Generated {len(orders_to_submit)} rebalancing orders.")

        # [Task 5.2] Concurrent Order Submission
        if orders_to_submit:
            await self._submit_orders_concurrently(orders_to_submit, market_prices)

    async def _submit_orders_concurrently(self, orders: List[Order], market_prices: Dict[str, Decimal]):
        """[Task 5.2] Helper to submit a batch of orders concurrently."""
        
        async def _submit(order: Order):
            price = float(order.limit_price) if order.limit_price else float(market_prices.get(order.symbol, 0))
            try:
                if self.broker:
                    # Convert Decimal fields to float for broker API if needed
                    logger.info(f"{self.log_prefix} Submitting {order.side} {order.quantity} {order.symbol}...")
                    await asyncio.to_thread(self.broker.place_order, order, price)
                
                async with self.lock:
                    current = self.active_orders.get(order.id)
                    if current and current.status == OrderStatus.PENDING_SUBMISSION:
                        current.status = OrderStatus.PENDING
            except Exception as e:
                logger.error(f"{self.log_prefix} Failed to place order {order.id}: {e}")
                async with self.lock:
                    order.status = OrderStatus.REJECTED
                    if order.id in self.active_orders: del self.active_orders[order.id]
        
        # Gather all submission tasks
        await asyncio.gather(*[_submit(o) for o in orders])

    async def cancel_all_orders(self):
        """取消所有活跃订单。"""
        async with self.lock:
            orders = list(self.active_orders.values())
        
        for order in orders:
            try:
                if self.broker:
                    await asyncio.to_thread(self.broker.cancel_order, order.id)
                async with self.lock:
                    order.status = OrderStatus.CANCELLED
                    if order.id in self.active_orders: del self.active_orders[order.id]
            except Exception as e:
                logger.error(f"{self.log_prefix} Failed to cancel order {order.id}: {e}")

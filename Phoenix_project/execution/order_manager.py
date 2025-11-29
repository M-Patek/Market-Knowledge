"""
订单管理器 (Order Manager)
负责处理订单的整个生命周期：发送、监控、取消。
[Beta Final Fix] 集成 Fail-Safe 逻辑，捕获数据完整性错误并执行紧急停止。
[Task 4] Concurrency Safety & Execution Protection
[Phase IV Fix] Time Machine Support & Precision/Risk Controls
"""
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from collections import defaultdict
import uuid
import asyncio
from .interfaces import IBrokerAdapter

# [阶段 1 & 4 变更] 导入依赖
from Phoenix_project.execution.trade_lifecycle_manager import TradeLifecycleManager
from Phoenix_project.data_manager import DataManager
from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.core.schemas.data_schema import (
    Order, Fill, OrderStatus, PortfolioState, TargetPortfolio, Position
)
from Phoenix_project.core.exceptions import RiskViolationError
from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

class OrderManager:
    """
    作为订单和券商适配器之间的中介。
    跟踪所有活动订单的状态。
    """
    
    def __init__(self, broker: IBrokerAdapter, trade_lifecycle_manager: TradeLifecycleManager, data_manager: DataManager):
        self.broker = broker
        self.trade_lifecycle_manager = trade_lifecycle_manager 
        self.data_manager = data_manager 
        self.active_orders: Dict[str, Order] = {} # key: order_id
        self.lock = asyncio.Lock() 
        
        self.valid_transitions = {
            OrderStatus.NEW: {OrderStatus.PENDING_SUBMISSION, OrderStatus.REJECTED, OrderStatus.CANCELLED},
            OrderStatus.PENDING_SUBMISSION: {OrderStatus.PENDING, OrderStatus.FILLED, OrderStatus.REJECTED, OrderStatus.CANCELLED},
            OrderStatus.PENDING: {OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED},
            OrderStatus.PARTIALLY_FILLED: {OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED},
        }
        
        self.max_order_value = 100000.0  
        self.max_notional_exposure = 1000000.0 
        
        if self.broker:
            self.broker.subscribe_fills(self._on_fill)
            self.broker.subscribe_order_status(self._on_order_status_update)
        
        self.log_prefix = "OrderManager:"
        logger.info(f"{self.log_prefix} Initialized.")

    async def _on_fill(self, fill: Fill):
        async with self.lock:
            logger.info(f"{self.log_prefix} Received Fill: {fill.order_id} ({fill.quantity} @ {fill.price})")
            try:
                await self.trade_lifecycle_manager.on_fill(fill)
            except Exception as e:
                logger.error(f"{self.log_prefix} Failed to process fill with TradeLifecycleManager: {e}", exc_info=True)

    async def _on_order_status_update(self, order: Order):
        async with self.lock:
            if not order or not order.id or not order.status:
                return

            current_order = self.active_orders.get(order.id)
            if current_order and not self._can_transition(current_order.status, order.status):
                logger.error(f"{self.log_prefix} INVALID STATE TRANSITION: {current_order.status} -> {order.status}")
                return
                
            logger.info(f"{self.log_prefix} Status Update: Order {order.id} is now {order.status}")
            
            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
                if order.id in self.active_orders:
                    del self.active_orders[order.id]
            else:
                self.active_orders[order.id] = order

    def _can_transition(self, current_status: str, new_status: str) -> bool:
        if current_status == new_status: return True
        allowed = self.valid_transitions.get(current_status, set())
        return new_status in allowed

    def _get_instrument_metadata(self, symbol: str) -> Dict[str, float]:
        return {"tick_size": 0.01, "lot_size": 1.0}

    def _round_to_tick(self, price: float, tick_size: float) -> float:
        """[Task 4] Fixes Penny Gap using Decimal arithmetic."""
        d_price = Decimal(str(price))
        d_tick = Decimal(str(tick_size))
        rounded = d_price.quantize(d_tick, rounding=ROUND_DOWN)
        return float(rounded)

    def generate_orders(self, 
                        current_portfolio: PortfolioState, 
                        target_portfolio: TargetPortfolio, 
                        market_prices: Dict[str, float],
                        active_orders: Dict[str, Order]
                       ) -> List[Order]:
        """
        [Task 4] Generates LIMIT orders based on effective position (settled + active).
        """
        logger.info(f"{self.log_prefix} Generating orders...")
        orders_to_generate: List[Order] = []
        
        if not current_portfolio or not target_portfolio: return []

        total_value = float(current_portfolio.total_value)
        if total_value <= 0:
            if float(current_portfolio.cash) > 0:
                 total_value = float(current_portfolio.cash)
            else:
                return []

        current_gross_exposure = sum(abs(float(p.market_value)) for p in current_portfolio.positions.values())

        # 1. Effective Position Calculation
        effective_quantities = defaultdict(float)
        for symbol, position in current_portfolio.positions.items():
            effective_quantities[symbol] += float(position.quantity)
        for order in active_orders.values():
            effective_quantities[order.symbol] += float(order.quantity)
            
        current_weights: Dict[str, float] = {}
        for symbol, qty in effective_quantities.items():
            price = market_prices.get(symbol, 0)
            if price > 0 and total_value != 0:
                 current_weights[symbol] = (qty * price) / total_value

        target_weights: Dict[str, float] = {}
        for pos in target_portfolio.positions:
            target_weights[pos.symbol] = float(pos.target_weight)

        all_symbols = set(current_weights.keys()) | set(target_weights.keys())

        for symbol in all_symbols:
            current_w = current_weights.get(symbol, 0.0)
            target_w = target_weights.get(symbol, 0.0)
            
            delta_weight = target_w - current_w
            
            if abs(delta_weight) > 0.0001:
                target_dollar_value = delta_weight * total_value
                price = market_prices.get(symbol)
                
                if price is None or price <= 0: continue
                
                order_quantity = target_dollar_value / price
                slippage_factor = 0.005
                limit_price = price * (1 + slippage_factor) if order_quantity > 0 else price * (1 - slippage_factor)
                
                meta = self._get_instrument_metadata(symbol)
                limit_price = self._round_to_tick(limit_price, meta["tick_size"])

                expected_value = order_quantity * price
                if abs(expected_value) > self.max_order_value:
                    raise RiskViolationError(f"Order value {expected_value:.2f} exceeds limit {self.max_order_value}.")
                
                new_order = Order(
                    id=f"order_{symbol}_{uuid.uuid4()}",
                    symbol=symbol,
                    quantity=Decimal(str(order_quantity)), # Convert to Decimal
                    price=Decimal(str(limit_price)),
                    order_type="LIMIT",
                    status=OrderStatus.NEW,
                    time_in_force="DAY",
                    metadata={"target_weight": target_w, "current_weight": current_w}
                )
                orders_to_generate.append(new_order)

        return orders_to_generate

    async def reconcile_portfolio(self, state: PipelineState):
        """[Task 4.2] Main Reconciliation Loop."""
        if state.l3_decision:
            risk_action = state.l3_decision.get("risk_action")
            if risk_action == "HALT_TRADING":
                logger.critical(f"{self.log_prefix} EMERGENCY BRAKE TRIGGERED. Cancelling all orders.")
                self.cancel_all_open_orders()
                return

        target_portfolio = state.target_portfolio
        if not target_portfolio: return

        symbols = [p.symbol for p in target_portfolio.positions]
        market_prices = {}
        
        for sym in symbols:
            md = await self.data_manager.get_latest_market_data(sym)
            if not md: continue
            
            time_diff = state.current_time - md.timestamp.replace(tzinfo=None) 
            if abs(time_diff) > timedelta(minutes=1): continue
                
            market_prices[sym] = float(md.close)
            
        await self._reconcile_active_orders()

        try:
            current_portfolio = self.trade_lifecycle_manager.get_current_portfolio_state(
                market_prices, timestamp=state.current_time
            )
        except ValueError as e:
            logger.critical(f"{self.log_prefix} DATA INTEGRITY ALERT: {e}. EMERGENCY STOP.")
            self.cancel_all_open_orders()
            return 

        async with self.lock:
            active_orders_snapshot = self.active_orders.copy()
            
        try:
            orders = self.generate_orders(current_portfolio, target_portfolio, market_prices, active_orders_snapshot)
        except RiskViolationError as e:
            logger.critical(f"{self.log_prefix} FATAL RISK VIOLATION: {e}. STOPPING ALL TRADING.")
            self.cancel_all_open_orders()
            return
        
        orders_to_submit = []
        async with self.lock:
            for order in orders:
                if order.id in self.active_orders: continue
                order.status = OrderStatus.PENDING_SUBMISSION
                self.active_orders[order.id] = order
                orders_to_submit.append(order)

        for order in orders_to_submit:
            price = float(order.limit_price) if order.limit_price else market_prices.get(order.symbol)
            try:
                if self.broker:
                    # Convert Decimal fields to float for broker API if needed, usually managed by adapter
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

    async def _reconcile_active_orders(self):
        """[Task 3.2] Debounce Logic: 3 Strikes Rule."""
        if not self.broker: return

        try:
            broker_open_orders = await asyncio.to_thread(self.broker.get_open_orders)
            broker_order_ids = {o.id for o in broker_open_orders}
            
            async with self.lock:
                local_ids = list(self.active_orders.keys())
                for oid in local_ids:
                    order = self.active_orders[oid]
                    if order.status in {OrderStatus.NEW, OrderStatus.PENDING_SUBMISSION}:
                        continue
                        
                    if oid not in broker_order_ids:
                        # [Task 3.2 Fix] Debounce logic
                        count = order.metadata.get('missing_count', 0) + 1
                        order.metadata['missing_count'] = count
                        
                        if count < 3:
                            logger.warning(f"{self.log_prefix} Order {oid} missing (Attempt {count}/3). Retaining.")
                        else:
                            logger.error(f"{self.log_prefix} Zombie Order Confirmed: {oid}. REJECTED.")
                            order.status = OrderStatus.REJECTED
                            del self.active_orders[oid]
                    else:
                        if order.metadata.get('missing_count', 0) > 0:
                            order.metadata['missing_count'] = 0
                        
        except Exception as e:
            logger.error(f"{self.log_prefix} Failed to reconcile active orders: {e}")

    def cancel_all_open_orders(self):
        order_ids = list(self.active_orders.keys())
        for order_id in order_ids:
            try:
                if self.broker: self.broker.cancel_order(order_id)
            except Exception: pass

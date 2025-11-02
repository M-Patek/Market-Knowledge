"""
订单管理器 (Order Manager)
负责处理订单的整个生命周期：发送、监控、取消。
"""
from typing import List, Dict, Optional
from .interfaces import IBrokerAdapter
import threading

# FIX (E2, E4): 从核心模式导入 Order, Fill, OrderStatus
from core.schemas.data_schema import Order, Fill, OrderStatus

from monitor.logging import get_logger

logger = get_logger(__name__)

class OrderManager:
    """
    作为订单和券商适配器之间的中介。
    跟踪所有活动订单的状态。
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

    def submit_orders(self, orders: List[Order]):
        """
        向券商提交一批新订单。
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

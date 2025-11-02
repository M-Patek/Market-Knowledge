"""
Order Manager
- 接收 StrategySignal (目标投资组合)
- 计算与当前持仓的差异
- 生成并发送订单到 Broker Adapter
- 跟踪订单状态
"""
from typing import Dict, List
from monitor.logging import get_logger
from .interfaces import IOrderManager, IBrokerAdapter, Order, Fill
from .signal_protocol import StrategySignal

class OrderManager(IOrderManager):
    """
    管理订单的生命周期，从信号生成到执行。
    """
    
    # 关键修正 (Error 4):
    # 构造函数现在接受一个 Broker Adapter (依赖注入)
    # 而不是在内部硬编码或不接受 adapter
    def __init__(self, config: Dict, adapter: IBrokerAdapter):
        self.config = config
        self.adapter = adapter  # 存储注入的 adapter
        self.logger = get_logger(self.__class__.__name__)
        self.current_positions: Dict[str, float] = {} # (应从 adapter 加载)
        self.active_orders: Dict[str, Order] = {}
        self.logger.info(f"OrderManager initialized with adapter: {adapter.__class__.__name__}")

    async def load_current_positions(self):
        """从 Broker Adapter 加载当前持仓"""
        try:
            # (假设 adapter 有一个 get_positions 方法)
            self.current_positions = await self.adapter.get_positions()
            self.logger.info(f"Loaded positions: {self.current_positions}")
        except Exception as e:
            self.logger.error(f"Failed to load positions: {e}")

    async def generate_orders_from_signal(self, signal: StrategySignal):
        """
        根据信号(目标权重)和当前持仓计算并生成订单。
        """
        self.logger.info(f"Received signal: {signal.strategy_id} | Targets: {signal.target_weights}")
        
        # (确保持仓是最新的)
        await self.load_current_positions()

        # (获取总资产净值 - 假设)
        total_equity = await self.adapter.get_total_equity() 
        
        orders_to_place: List[Order] = []
        target_weights = signal.target_weights
        
        # (这是一个简化的差分逻辑)
        # 1. 计算所有标的的目标美元价值
        all_tickers = set(target_weights.keys()) | set(self.current_positions.keys())

        for ticker in all_tickers:
            target_weight = target_weights.get(ticker, 0.0)
            target_value = total_equity * target_weight
            
            current_value = self.current_positions.get(ticker, {}).get('market_value', 0.0)
            
            delta_value = target_value - current_value
            
            # (应用一个阈值，避免小额交易)
            if abs(delta_value) > self.config.get('min_trade_value', 100): 
                # (需要从 $ 转换到 数量 qty)
                # (此处需要价格数据... 简化处理)
                try:
                    # (假设 adapter 有 get_last_price 方法)
                    last_price = await self.adapter.get_last_price(ticker)
                    if last_price <= 0:
                        raise ValueError("Price is zero or negative")
                        
                    qty = delta_value / last_price
                    
                    order = Order(
                        ticker=ticker,
                        qty=abs(qty),
                        side="buy" if delta_value > 0 else "sell",
                        order_type="market", # (或 "limit")
                        # (limit_price=...)
                    )
                    orders_to_place.append(order)
                    
                except Exception as e:
                    self.logger.error(f"Could not create order for {ticker}: {e}")

        # 2. 发送订单
        await self.place_orders(orders_to_place)

    async def place_orders(self, orders: List[Order]):
        """将订单列表发送到 broker adapter"""
        for order in orders:
            try:
                order_id = await self.adapter.place_order(order)
                order.broker_order_id = order_id
                self.active_orders[order_id] = order
                self.logger.info(f"Placed order: {order.side} {order.qty} {order.ticker} | ID: {order_id}")
            except Exception as e:
                self.logger.error(f"Failed to place order for {order.ticker}: {e}")

    async def on_fill(self, fill_event: Fill):
        """
        处理来自 adapter 的订单成交事件。
        """
        self.logger.info(f"Received fill: {fill_event.status} {fill_event.filled_qty} {fill_event.ticker} @ {fill_event.avg_fill_price}")
        # (更新持仓和活动订单列表)
        if fill_event.order_id in self.active_orders:
            if fill_event.status == "filled" or fill_event.status == "cancelled":
                del self.active_orders[fill_event.order_id]
        
        # (需要更新 self.current_positions)
        await self.load_current_positions() 

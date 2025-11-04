"""
交易生命周期管理器 (Trade Lifecycle Manager)
负责跟踪从信号 -> 订单 -> 成交 -> 持仓 的整个过程。
计算已实现和未实现的盈亏 (PnL)。
"""
from typing import Dict
from datetime import datetime

# FIX (E2, E4): 从核心模式导入 Order, Fill, Position, PortfolioState
# 修正：将 'core.schemas...' 转换为 'Phoenix_project.core.schemas...'
from Phoenix_project.core.schemas.data_schema import Order, Fill, Position, PortfolioState

# 修正：将 'monitor.logging...' 转换为 'Phoenix_project.monitor.logging...'
from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

class TradeLifecycleManager:
    """
    维护投资组合的当前状态 (持仓和现金)。
    """
    def __init__(self, initial_cash: float):
        self.positions: Dict[str, Position] = {} # key: symbol
        self.cash = initial_cash
        self.realized_pnl = 0.0
        self.log_prefix = "TradeLifecycleManager:"
        logger.info(f"{self.log_prefix} Initialized with initial cash: {initial_cash}")

    def get_current_portfolio_state(self, current_market_data: Dict[str, float]) -> PortfolioState:
        """
        根据最新的市场价格计算并返回当前的投资组合状态。
        :param current_market_data: Dict[symbol, current_price]
        """
        total_value = self.cash
        
        # 更新持仓的市值和未实现盈亏
        for symbol, pos in self.positions.items():
            current_price = current_market_data.get(symbol)
            if current_price:
                pos.market_value = pos.quantity * current_price
                pos.unrealized_pnl = (current_price - pos.average_price) * pos.quantity
                total_value += pos.market_value
            else:
                logger.warning(f"{self.log_prefix} Missing market data for {symbol} to update PnL.")
                # 使用上一次的市值
                total_value += pos.market_value
                
        return PortfolioState(
            timestamp=datetime.utcnow(), # 实际应使用事件时间
            cash=self.cash,
            total_value=total_value,
            positions=self.positions.copy(),
            realized_pnl=self.realized_pnl
        )

    def on_fill(self, fill: Fill):
        """
        核心逻辑：当收到成交回报时，更新持仓和现金。
        """
        logger.info(f"{self.log_prefix} Processing fill for {fill.symbol}: {fill.quantity} @ {fill.price}")
        
        # 1. 更新现金
        trade_cost = fill.price * fill.quantity
        self.cash -= trade_cost
        self.cash -= fill.commission
        
        # 2. 更新持仓
        current_pos = self.positions.get(
            fill.symbol,
            Position(symbol=fill.symbol, quantity=0.0, average_price=0.0, market_value=0.0, unrealized_pnl=0.0)
        )
        
        current_qty = current_pos.quantity
        current_avg_price = current_pos.average_price
        
        new_qty = current_qty + fill.quantity
        
        if abs(new_qty) < 1e-6:
            # 仓位已平仓
            logger.info(f"{self.log_prefix} Position closed for {fill.symbol}")
            # 计算已实现 PnL
            pnl = (fill.price - current_avg_price) * (-current_qty) # -current_qty 是平仓的数量
            self.realized_pnl += pnl
            del self.positions[fill.symbol]
            
        elif current_qty * fill.quantity >= 0: 
            # 增加仓位 (同向交易)
            new_avg_price = ((current_avg_price * current_qty) + (fill.price * fill.quantity)) / new_qty
            
            current_pos.quantity = new_qty
            current_pos.average_price = new_avg_price
            self.positions[fill.symbol] = current_pos
            logger.info(f"{self.log_prefix} Position updated for {fill.symbol}: New Qty={new_qty}, New AvgPx={new_avg_price}")

        else:
            # 减少仓位或反转仓位 (异向交易)
            if abs(fill.quantity) <= abs(current_qty):
                # 减少仓位
                pnl = (fill.price - current_avg_price) * abs(fill.quantity)
                self.realized_pnl += pnl
                
                current_pos.quantity = new_qty
                # 平均价格不变
                self.positions[fill.symbol] = current_pos
                logger.info(f"{self.log_prefix} Position reduced for {fill.symbol}: New Qty={new_qty}, Realized PnL={pnl}")
            else:
                # 反转仓位 (e.g., 从 +100 到 -50)
                # 1. 平掉所有旧仓位
                pnl = (fill.price - current_avg_price) * abs(current_qty)
                self.realized_pnl += pnl
                
                # 2. 建立新仓位
                current_pos.quantity = new_qty
                current_pos.average_price = fill.price # 新仓位的成本价是当前成交价
                self.positions[fill.symbol] = current_pos
                logger.info(f"{self.log_prefix} Position reversed for {fill.symbol}: New Qty={new_qty}, Realized PnL={pnl}")

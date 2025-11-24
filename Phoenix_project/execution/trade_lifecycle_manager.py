"""
交易生命周期管理器 (Trade Lifecycle Manager)
负责跟踪从信号 -> 订单 -> 成交 -> 持仓 的整个过程。
计算已实现和未实现的盈亏 (PnL)。
[Beta FIX] 防止僵尸估值 (Zombie Valuation)
[Task 3] Atomic Transactions & Write-Ahead Persistence
[Phase III Fix] Valuation Resilience (Limp Mode)
"""
from typing import Dict, Optional
from datetime import datetime
import asyncio

# FIX (E2, E4): 从核心模式导入 Order, Fill, Position, PortfolioState
from Phoenix_project.core.schemas.data_schema import Order, Fill, Position, PortfolioState

# 修正：将 'monitor.logging...' 转换为 'Phoenix_project.monitor.logging...'
from Phoenix_project.monitor.logging import get_logger
from Phoenix_project.ai.tabular_db_client import TabularDBClient

logger = get_logger(__name__)

class TradeLifecycleManager:
    """
    [Refactored Phase 4.1] 持久化账本管理器。
    维护投资组合状态 (持仓和现金)，并同步到 TabularDB 以防止状态丢失。
    """
    def __init__(self, initial_cash: float, tabular_db: Optional[TabularDBClient] = None):
        self.positions: Dict[str, Position] = {} # key: symbol
        self.cash = initial_cash
        self.realized_pnl = 0.0
        self.tabular_db = tabular_db
        self.log_prefix = "TradeLifecycleManager:"
        self._lock = asyncio.Lock()
        logger.info(f"{self.log_prefix} Initialized. DB Connected: {self.tabular_db is not None}")

    async def initialize(self):
        """
        [Task 4.1] 初始化账本。
        1. 确保数据库表存在。
        2. 从数据库加载最新的资金和持仓状态 (恢复现场)。
        """
        if not self.tabular_db:
            logger.warning(f"{self.log_prefix} No TabularDB configured. Running in IN-MEMORY mode (Volatile!).")
            return

        try:
            logger.info(f"{self.log_prefix} initializing persistent ledger...")
            # 1. 建表 (如果不存在)
            await self._ensure_tables_exist()

            # 2. 加载状态
            await self._load_ledger()
            
        except Exception as e:
            logger.critical(f"{self.log_prefix} Failed to initialize ledger: {e}", exc_info=True)
            raise

    async def _ensure_tables_exist(self):
        """创建 ledger_balance, ledger_positions, ledger_fills 表。"""
        # Balance Table
        await self.tabular_db.query("""
            CREATE TABLE IF NOT EXISTS ledger_balance (
                id SERIAL PRIMARY KEY,
                cash FLOAT NOT NULL,
                realized_pnl FLOAT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        # Positions Table
        await self.tabular_db.query("""
            CREATE TABLE IF NOT EXISTS ledger_positions (
                symbol VARCHAR(20) PRIMARY KEY,
                quantity FLOAT NOT NULL,
                average_price FLOAT NOT NULL,
                market_value FLOAT NOT NULL,
                unrealized_pnl FLOAT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        # Fills Table (For Idempotency)
        await self.tabular_db.query("""
            CREATE TABLE IF NOT EXISTS ledger_fills (
                fill_id VARCHAR(64) PRIMARY KEY,
                order_id VARCHAR(64),
                symbol VARCHAR(20),
                quantity FLOAT,
                price FLOAT,
                timestamp TIMESTAMP
            );
        """)

    async def _load_ledger(self):
        """从数据库加载状态。"""
        # Load Balance
        res = await self.tabular_db.query("SELECT cash, realized_pnl FROM ledger_balance ORDER BY id DESC LIMIT 1")
        if res and "results" in res and res["results"]:
            row = res["results"][0]
            self.cash = float(row["cash"])
            self.realized_pnl = float(row["realized_pnl"])
            logger.info(f"{self.log_prefix} Restored balance: Cash={self.cash}, PnL={self.realized_pnl}")

        # Load Positions
        res_pos = await self.tabular_db.query("SELECT * FROM ledger_positions")
        if res_pos and "results" in res_pos:
            for row in res_pos["results"]:
                sym = row["symbol"]
                self.positions[sym] = Position(
                    symbol=sym,
                    quantity=float(row["quantity"]),
                    average_price=float(row["average_price"]),
                    market_value=float(row["market_value"]),
                    unrealized_pnl=float(row["unrealized_pnl"])
                )
            logger.info(f"{self.log_prefix} Restored {len(self.positions)} positions.")

    def get_current_portfolio_state(self, current_market_data: Dict[str, float]) -> PortfolioState:
        """
        根据最新的市场价格计算并返回当前的投资组合状态。
        [Beta FIX] 增加了严格的数据完整性检查。
        [Phase III Fix] Valuation Resilience (Limp Mode)
        :param current_market_data: Dict[symbol, current_price]
        """
        total_value = self.cash
        
        # 更新持仓的市值和未实现盈亏
        for symbol, pos in self.positions.items():
            current_price = current_market_data.get(symbol)
            
            # [Phase III Fix] Resilience: Warn and use fallback (Cost Basis) instead of crashing
            # 如果价格缺失或非正数，降级为警告，并使用成本价估值
            if current_price is None or current_price <= 0:
                logger.warning(f"{self.log_prefix} Missing/Invalid price for {symbol}. Using Cost Basis (Avg Price) as fallback.")
                current_price = pos.average_price

            pos.market_value = pos.quantity * current_price
            pos.unrealized_pnl = (current_price - pos.average_price) * pos.quantity
            total_value += pos.market_value
                
        return PortfolioState(
            timestamp=datetime.utcnow(), # 实际应使用事件时间
            cash=self.cash,
            total_value=total_value,
            positions=self.positions.copy(),
            realized_pnl=self.realized_pnl
        )

    async def on_fill(self, fill: Fill):
        """
        [Task 4.1] 异步处理 Fill 事件。
        包含幂等性检查和持久化。
        """
        async with self._lock:
            # 1. 幂等性检查
            if await self._is_fill_processed(fill.id):
                logger.warning(f"{self.log_prefix} Skipping duplicate fill {fill.id}")
                return

            logger.info(f"{self.log_prefix} Processing fill for {fill.symbol}: {fill.quantity} @ {fill.price}")
            
            # [Task 3] Atomic Transaction: Phase 1 - Pre-calculate new state (Shadow State)
            # Do NOT modify self attributes yet.
            trade_cost = fill.price * fill.quantity
            new_cash = self.cash - trade_cost - fill.commission
            new_realized_pnl = self.realized_pnl
            
            # 3. 计算新持仓状态
            current_pos = self.positions.get(
                fill.symbol,
                Position(symbol=fill.symbol, quantity=0.0, average_price=0.0, market_value=0.0, unrealized_pnl=0.0)
            )
            
            current_qty = current_pos.quantity
            current_avg_price = current_pos.average_price
            
            new_qty = current_qty + fill.quantity
            new_pos: Optional[Position] = None
            
            if abs(new_qty) < 1e-6:
                # 仓位已平仓
                logger.info(f"{self.log_prefix} Position closed for {fill.symbol}")
                # 计算已实现 PnL (Fix: Direction specific)
                qty_closed = abs(current_qty)
                if current_qty > 0:
                    pnl = (fill.price - current_avg_price) * qty_closed
                else:
                    pnl = (current_avg_price - fill.price) * qty_closed

                new_realized_pnl += pnl
                new_pos = None # Mark for deletion
                
            elif current_qty * fill.quantity >= 0: 
                # 增加仓位 (同向交易)
                new_avg_price = ((current_avg_price * current_qty) + (fill.price * fill.quantity)) / new_qty
                
                # Create new position object (immutable style preferred)
                new_pos = current_pos.model_copy()
                new_pos.quantity = new_qty
                new_pos.average_price = new_avg_price
                logger.info(f"{self.log_prefix} Position updated for {fill.symbol}: New Qty={new_qty}, New AvgPx={new_avg_price}")

            else:
                # 减少仓位或反转仓位 (异向交易)
                if abs(fill.quantity) <= abs(current_qty):
                    # 减少仓位
                    qty_closed = abs(fill.quantity)
                    if current_qty > 0:
                        pnl = (fill.price - current_avg_price) * qty_closed
                    else:
                        pnl = (current_avg_price - fill.price) * qty_closed
                    
                    new_realized_pnl += pnl
                    
                    new_pos = current_pos.model_copy()
                    new_pos.quantity = new_qty
                    # 平均价格不变
                    logger.info(f"{self.log_prefix} Position reduced for {fill.symbol}: New Qty={new_qty}, Realized PnL={pnl}")
                else:
                    # 反转仓位 (e.g., 从 +100 到 -50)
                    # 1. 平掉所有旧仓位
                    qty_closed = abs(current_qty)
                    if current_qty > 0:
                        pnl = (fill.price - current_avg_price) * qty_closed
                    else:
                        pnl = (current_avg_price - fill.price) * qty_closed
                    
                    new_realized_pnl += pnl
                    
                    # 2. 建立新仓位
                    new_pos = current_pos.model_copy()
                    new_pos.quantity = new_qty
                    new_pos.average_price = fill.price # 新仓位的成本价是当前成交价
                    logger.info(f"{self.log_prefix} Position reversed for {fill.symbol}: New Qty={new_qty}, Realized PnL={pnl}")

            # [Task 3] Atomic Transaction: Phase 2 - DB Persistence (Write-Ahead)
            if self.tabular_db:
                try:
                    # Pass EXPLICIT calculated values, not self.*
                    await self._persist_transaction(new_cash, new_realized_pnl, new_pos, fill.symbol)
                    await self._record_fill(fill)
                except Exception as e:
                    # CRITICAL: DB write failed. Do NOT update memory. Halt system.
                    logger.critical(f"{self.log_prefix} LEDGER WRITE FAILED. HALTING to prevent split-brain. Error: {e}")
                    raise

            # [Task 3] Atomic Transaction: Phase 3 - Memory Commit
            # Only reached if DB write succeeded (or if DB is disabled)
            self.cash = new_cash
            self.realized_pnl = new_realized_pnl
            
            if new_pos:
                self.positions[fill.symbol] = new_pos
            else:
                # Safe removal
                self.positions.pop(fill.symbol, None)

    async def _is_fill_processed(self, fill_id: str) -> bool:
        """检查 Fill ID 是否已存在于数据库。"""
        if not self.tabular_db: return False
        res = await self.tabular_db.query(f"SELECT fill_id FROM ledger_fills WHERE fill_id = '{fill_id}'")
        return bool(res and "results" in res and res["results"])

    async def _persist_transaction(self, cash: float, realized_pnl: float, position: Optional[Position], symbol: str):
        """[Task 3] Atomic DB Update Helper. Uses explicit values."""
        # Update Balance
        await self.tabular_db.upsert_data(
            "ledger_balance", 
            {"id": 1, "cash": cash, "realized_pnl": realized_pnl}, 
            "id"
        )
        
        # Update Position (if exists)
        if position:
            await self.tabular_db.upsert_data(
                "ledger_positions",
                position.model_dump(),
                "symbol"
            )
        else:
            # Position closed, remove from DB
            await self.tabular_db.query(f"DELETE FROM ledger_positions WHERE symbol = '{symbol}'")

    async def _record_fill(self, fill: Fill):
        """记录 Fill 事件以防止重放。"""
        await self.tabular_db.upsert_data(
            "ledger_fills",
            {"fill_id": fill.id, "order_id": fill.order_id, "symbol": fill.symbol, 
             "quantity": fill.quantity, "price": fill.price, "timestamp": fill.timestamp},
            "fill_id"
        )

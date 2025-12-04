"""
交易生命周期管理器 (Trade Lifecycle Manager)
负责跟踪从信号 -> 订单 -> 成交 -> 持仓 的整个过程。
计算已实现和未实现的盈亏 (PnL)。
[Beta FIX] 防止僵尸估值 (Zombie Valuation) & 脏读崩溃 (Dirty Read)
[Task 3] Atomic Transactions & Write-Ahead Persistence
[Phase III Fix] Valuation Resilience (Fail-Closed)
[Phase II Fix] Decimal Precision & Atomic Transactions
[Phase V Fix] SQL Injection Protection
[Code Opt Expert Fix] Task 11 & 12: Concurrency Crash Fix & Valuation Fallback
"""
from typing import Dict, Optional
from datetime import datetime
from decimal import Decimal
import asyncio
from sqlalchemy.exc import IntegrityError

# FIX (E2, E4): 从核心模式导入 Order, Fill, Position, PortfolioState
from Phoenix_project.core.schemas.data_schema import Order, Fill, Position, PortfolioState

from Phoenix_project.monitor.logging import get_logger
from Phoenix_project.ai.tabular_db_client import TabularDBClient

logger = get_logger(__name__)

class TradeLifecycleManager:
    """
    [Refactored Phase 4.1] 持久化账本管理器。
    维护投资组合状态 (持仓和现金)，并同步到 TabularDB 以防止状态丢失。
    [Phase II Fix] Decimal Precision & Atomic Transactions
    """
    def __init__(self, initial_cash: float, tabular_db: Optional[TabularDBClient] = None):
        self.positions: Dict[str, Position] = {} # key: symbol
        # [Phase II Fix] Use Decimal for financial precision (The "Penny Gap")
        self.cash = Decimal(str(initial_cash))
        self.realized_pnl = Decimal("0.0")
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
                cash NUMERIC(20, 10) NOT NULL,
                realized_pnl NUMERIC(20, 10) NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        # Positions Table
        await self.tabular_db.query("""
            CREATE TABLE IF NOT EXISTS ledger_positions (
                symbol VARCHAR(20) PRIMARY KEY,
                quantity NUMERIC(20, 10) NOT NULL,
                average_price NUMERIC(20, 10) NOT NULL,
                market_value NUMERIC(20, 10) NOT NULL,
                unrealized_pnl NUMERIC(20, 10) NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        # Fills Table (For Idempotency)
        await self.tabular_db.query("""
            CREATE TABLE IF NOT EXISTS ledger_fills (
                fill_id VARCHAR(64) PRIMARY KEY,
                order_id VARCHAR(64),
                symbol VARCHAR(20),
                quantity NUMERIC(20, 10),
                price NUMERIC(20, 10),
                timestamp TIMESTAMP
            );
        """)

    async def _load_ledger(self):
        """从数据库加载状态。"""
        # Load Balance
        res = await self.tabular_db.query("SELECT cash, realized_pnl FROM ledger_balance ORDER BY id DESC LIMIT 1")
        if res and "results" in res and res["results"]:
            row = res["results"][0]
            # [Phase II Fix] Restore as Decimal
            self.cash = Decimal(str(row["cash"]))
            self.realized_pnl = Decimal(str(row["realized_pnl"]))
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

    def get_current_portfolio_state(self, current_market_data: Dict[str, float], timestamp: Optional[datetime] = None) -> PortfolioState:
        """
        根据最新的市场价格计算并返回当前的投资组合状态。
        [Beta FIX] 增加了严格的数据完整性检查 (Ostrich Valuation Fix)。
        [Fix] 增加了线程安全的迭代快照 (Dirty Read Fix)。
        :param current_market_data: Dict[symbol, current_price]
        :param timestamp: Optional simulation timestamp (for backtesting consistency)
        """
        # [Fix Dirty Read] Create a shallow copy of positions to allow safe iteration 
        # while on_fill might be modifying the main dict in the background.
        # [Task 11] Atomic Snapshot: Copy dict keys/refs first to prevent iteration errors
        raw_positions = self.positions.copy()
        positions_snapshot = {k: v.model_copy() for k, v in raw_positions.items()}
        
        total_value = self.cash
        
        # 更新持仓的市值和未实现盈亏 (使用快照)
        for symbol, pos in positions_snapshot.items():
            current_price = current_market_data.get(symbol)
            
            # [Task 2.3 Fix] Remove Ostrich Valuation (Fail-Closed)
            if current_price is None or current_price <= 0:
                raise ValueError(f"CRITICAL: Missing market data for {symbol}. Cannot value portfolio.")

            # Calculate in Decimal for safety
            # Note: Explicit casting Decimal -> float for schema compatibility
            try:
                # 使用 Decimal 进行高精度计算，避免浮点数漂移
                d_qty = Decimal(str(pos.quantity))
                d_price = Decimal(str(current_price))
                d_avg = Decimal(str(pos.average_price))
                
                pos.market_value = float(d_qty * d_price)
                pos.unrealized_pnl = float((d_price - d_avg) * d_qty)
                
                total_value += Decimal(str(pos.market_value))
            except Exception as e:
                logger.error(f"{self.log_prefix} Valuation error for {symbol}: {e}")
                # 保持原值或设为0，视具体策略。这里由上层捕获。
        
        # Determine timestamp: Use provided simulation time or UTC now (Default)
        # [Fix] Prefer injected timestamp for causal consistency
        state_timestamp = timestamp if timestamp else datetime.utcnow()

        return PortfolioState(
            timestamp=state_timestamp,
            cash=float(self.cash), # Convert back to float for Schema
            total_value=float(total_value),
            positions=positions_snapshot, # Return the consistent snapshot
            realized_pnl=float(self.realized_pnl)
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
            # Use Decimal for high precision arithmetic
            fill_qty = Decimal(str(fill.quantity))
            fill_price = Decimal(str(fill.price))
            fill_comm = Decimal(str(fill.commission))
            
            trade_cost = fill_price * fill_qty
            new_cash = self.cash - trade_cost - fill_comm
            new_realized_pnl = self.realized_pnl
            
            # 3. 计算新持仓状态
            current_pos = self.positions.get(
                fill.symbol,
                Position(symbol=fill.symbol, quantity=0.0, average_price=0.0, market_value=0.0, unrealized_pnl=0.0)
            )
            
            current_qty = Decimal(str(current_pos.quantity))
            current_avg_price = Decimal(str(current_pos.average_price))
            
            new_qty = current_qty + fill_qty
            new_pos: Optional[Position] = None
            
            if abs(new_qty) < Decimal("1e-6"):
                # 仓位已平仓
                logger.info(f"{self.log_prefix} Position closed for {fill.symbol}")
                # 计算已实现 PnL (Fix: Direction specific)
                qty_closed = abs(current_qty)
                # PnL = (Exit Price - Entry Price) * Qty * Direction
                if current_qty > 0: # Long Close
                    pnl = (fill_price - current_avg_price) * qty_closed
                else: # Short Close
                    pnl = (current_avg_price - fill_price) * qty_closed

                new_realized_pnl += pnl
                new_pos = None # Mark for deletion
                
            elif current_qty * Decimal(str(fill.quantity)) >= 0: 
                # 增加仓位 (同向交易)
                new_avg_price = ((current_avg_price * current_qty) + (fill_price * fill_qty)) / new_qty
                
                # Create new position object (immutable style preferred)
                new_pos = current_pos.model_copy()
                new_pos.quantity = float(new_qty)
                new_pos.average_price = float(new_avg_price)
                logger.info(f"{self.log_prefix} Position updated for {fill.symbol}: New Qty={new_qty}, New AvgPx={new_avg_price}")

            else:
                # 减少仓位或反转仓位 (异向交易)
                if abs(fill_qty) <= abs(current_qty):
                    # 减少仓位
                    qty_closed = abs(fill_qty)
                    if current_qty > 0:
                        pnl = (fill_price - current_avg_price) * qty_closed
                    else:
                        pnl = (current_avg_price - fill_price) * qty_closed
                    
                    new_realized_pnl += pnl
                    
                    new_pos = current_pos.model_copy()
                    new_pos.quantity = float(new_qty)
                    # 平均价格不变
                    logger.info(f"{self.log_prefix} Position reduced for {fill.symbol}: New Qty={new_qty}, Realized PnL={pnl}")
                else:
                    # 反转仓位 (e.g., 从 +100 到 -50)
                    # 1. 平掉所有旧仓位
                    qty_closed = abs(current_qty)
                    if current_qty > 0:
                        pnl = (fill_price - current_avg_price) * qty_closed
                    else:
                        pnl = (current_avg_price - fill_price) * qty_closed
                    
                    new_realized_pnl += pnl
                    
                    # 2. 建立新仓位
                    new_pos = current_pos.model_copy()
                    new_pos.quantity = float(new_qty)
                    new_pos.average_price = float(fill_price) # 新仓位的成本价是当前成交价
                    logger.info(f"{self.log_prefix} Position reversed for {fill.symbol}: New Qty={new_qty}, Realized PnL={pnl}")

            # [Task 3] Atomic Transaction: Phase 2 - DB Persistence (Write-Ahead)
            if self.tabular_db:
                try:
                    # [Phase II Fix] ATOMICITY: Wrap ledger and fill record in a single DB transaction
                    async with self.tabular_db.transaction() as conn:
                        await self._persist_transaction(new_cash, new_realized_pnl, new_pos, fill.symbol, conn)
                        await self._record_fill(fill, conn)
                except IntegrityError:
                    logger.warning(f"{self.log_prefix} Duplicate fill {fill.id} detected during DB commit. Rolling back.")
                    return # Graceful exit, memory state remains untouched
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
        # [Task 5.1] Security Fix: Use parameterized query via execute_sql instead of f-string
        res = await self.tabular_db.execute_sql(
            "SELECT fill_id FROM ledger_fills WHERE fill_id = :fill_id",
            {"fill_id": fill_id}
        )
        return bool(res)

    async def _persist_transaction(self, cash: Decimal, realized_pnl: Decimal, position: Optional[Position], symbol: str, conn=None):
        """[Task 3] Atomic DB Update Helper. Uses explicit values."""
        # Update Balance
        await self.tabular_db.upsert_data(
            "ledger_balance", 
            {"id": 1, "cash": str(cash), "realized_pnl": str(realized_pnl)}, 
            "id",
            connection=conn
        )
        
        # Update Position (if exists)
        if position:
            await self.tabular_db.upsert_data(
                "ledger_positions",
                position.model_dump(),
                "symbol",
                connection=conn
            )
        else:
            # Position closed, remove from DB
            # Use execute_sql with conn for atomic delete
            await self.tabular_db.execute_sql(f"DELETE FROM ledger_positions WHERE symbol = :symbol", {"symbol": symbol}, connection=conn)

    async def _record_fill(self, fill: Fill, conn=None):
        """记录 Fill 事件以防止重放。"""
        await self.tabular_db.insert_data(
            "ledger_fills",
            {"fill_id": fill.id, "order_id": fill.order_id, "symbol": fill.symbol, 
             "quantity": fill.quantity, "price": fill.price, "timestamp": fill.timestamp},
            "fill_id",
            connection=conn
        )

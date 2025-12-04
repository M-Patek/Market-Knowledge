"""
Backtest Engine
- 模拟执行交易策略
- 使用历史数据
- 评估策略表现
[Task 016, 019] Risk Integration & Short Selling Constraint
"""
from Phoenix_project.monitor.logging import get_logger
from Phoenix_project.data_manager import DataManager
from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.core.schemas.data_schema import PortfolioState, Position, MarketData
from Phoenix_project.cognitive.engine import CognitiveEngine
from Phoenix_project.core.exceptions import RiskViolationError, CircuitBreakerError
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from collections import defaultdict
import asyncio

class BacktestEngine:
    """
    用于 Walk-Forward 训练和评估的模拟引擎。
    [Phase II Fix] 集成 RiskManager 进行预交易检查，防止过拟合。
    """
    
    def __init__(
        self, 
        config: Dict[str, Any],
        data_manager: DataManager,
        pipeline_state: PipelineState,
        cognitive_engine: CognitiveEngine,
        portfolio_constructor: Any = None,
        order_manager: Any = None,
        risk_manager: Any = None # [Task 4.2 Fix] Inject RiskManager
    ):
        self.config = config
        self.data_manager = data_manager
        self.pipeline_state = pipeline_state
        self.cognitive_engine = cognitive_engine
        self.portfolio_constructor = portfolio_constructor
        self.order_manager = order_manager
        self.risk_manager = risk_manager
        
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info("BacktestEngine initialized with authentic simulation components.")
        
        self.equity_curve = []   # Track equity over time
        self.trade_log = []      # Track executed trades
        self.pending_orders = defaultdict(float) # Store orders for T+1 execution

    async def run_backtest(self, data_iterator):
        """
        [Real Implementation] Iterate through data, generate signals, execute trades, track equity.
        [Phase II Fix] Converted to async to support RiskManager.
        """
        self.logger.info("Starting authentic backtest run...")
        self.equity_curve = []
        self.trade_log = []
        self.pending_orders.clear()

        # Initialize portfolio state if not present
        if not self.pipeline_state.portfolio_state:
            initial_cash = self.config.get("initial_cash", 100000.0)
            self.pipeline_state.portfolio_state = PortfolioState(
                cash=initial_cash,
                positions={},
                total_value=initial_cash
            )

        for current_time, market_data_slice in data_iterator:
            # 1. Update Pipeline State Time and Data
            self.pipeline_state.update_time(current_time)
            # Set dynamic attribute for PortfolioConstructor access
            setattr(self.pipeline_state, 'market_data_batch', market_data_slice)

            # [Task 4.2] Step A: Execute Pending Orders from T-1 at Current Open (T)
            await self._execute_pending_orders(market_data_slice)

            # [Task 016] Backtest Risk Integration: Update RiskManager state
            # Ensure RiskManager sees the data AFTER execution (T) but BEFORE next decision (T -> T+1)
            if self.risk_manager:
                # Handle various data structures (List or Object)
                data_list = market_data_slice
                if hasattr(market_data_slice, 'market_data'):
                    data_list = market_data_slice.market_data
                elif isinstance(market_data_slice, dict) and 'market_data' in market_data_slice:
                    data_list = market_data_slice['market_data']
                
                # Iterate and update
                if isinstance(data_list, list):
                    for item in data_list:
                        # Convert dict to MarketData if needed, or use object
                        try:
                            md = item if isinstance(item, MarketData) else MarketData(**item)
                            await self.risk_manager.on_market_data(md)
                        except Exception as e:
                            self.logger.warning(f"Failed to update risk manager in backtest: {e}")

            # [Task 4.2] Step B: Generate Signals for T (to be executed at T+1)
            target_portfolio = None
            if self.portfolio_constructor:
                # Generate Target for *Next* Step
                target_portfolio = await self.portfolio_constructor.construct_portfolio(self.pipeline_state)
                if target_portfolio:
                    self._stage_orders_for_next_bar(target_portfolio, market_data_slice)
            
            # 4. Mark-to-Market (Update Equity)
            self._update_portfolio_valuation(market_data_slice)
            
            # 5. Record Metrics
            self.equity_curve.append({
                "timestamp": current_time,
                "equity": self.pipeline_state.portfolio_state.total_value,
                "cash": self.pipeline_state.portfolio_state.cash
            })

        self.logger.info("Backtest run completed.")
        return self.get_performance_metrics()

    def _stage_orders_for_next_bar(self, target_portfolio, market_data):
        """
        [Task 4.2] Calculate difference between Target and Current, store as Pending.
        Does NOT execute.
        """
        current_pf = self.pipeline_state.portfolio_state
        
        # 1. Map current prices (Close price of T is used for Signal Gen)
        price_map = self._extract_price_map(market_data, price_type='close')
        
        for pos in target_portfolio.positions:
            symbol = pos.symbol
            price = price_map.get(symbol, 0.0)
            if price <= 0: continue

            target_val = current_pf.total_value * pos.target_weight
            
            current_qty = 0.0
            if symbol in current_pf.positions:
                current_qty = current_pf.positions[symbol].quantity
            
            target_qty = target_val / price
            qty_delta = target_qty - current_qty
            
            # Store for T+1 execution
            self.pending_orders[symbol] = qty_delta

    async def _execute_pending_orders(self, market_data):
        """
        [Task 4.2] Execute pending orders at the OPEN price of the current bar (T).
        Realism: Fills occur at the first available price of the new period.
        [Phase II Fix] Enforce Pre-Trade Risk Checks.
        """
        if not self.pending_orders:
            return
            
        current_pf = self.pipeline_state.portfolio_state
        price_map = self._extract_price_map(market_data, price_type='open') # Use OPEN
        
        COMMISSION_RATE = 0.001 
        SLIPPAGE_RATE = 0.0005  

        for symbol, qty_delta in list(self.pending_orders.items()):
            price = price_map.get(symbol, 0.0)
            if price <= 0: continue 
            
            # [Task 4.2 Fix] Pre-Trade Risk Check
            if self.risk_manager:
                try:
                    proposed_trade = Position(
                        symbol=symbol,
                        quantity=qty_delta, # Delta
                        average_price=price,
                        market_value=abs(qty_delta * price),
                        unrealized_pnl=0.0
                    )
                    # Check Risk
                    self.risk_manager.check_pre_trade(proposed_trade, current_pf)
                except (RiskViolationError, CircuitBreakerError) as e:
                    self.logger.warning(f"Risk Check Failed for {symbol}: {e}. Order REJECTED.")
                    continue

            # Execute
            if symbol not in current_pf.positions:
                current_pf.positions[symbol] = Position(symbol=symbol, quantity=0.0, average_price=price, market_value=0.0, unrealized_pnl=0.0)
            
            trade_value = abs(qty_delta * price)
            cost = trade_value * (COMMISSION_RATE + SLIPPAGE_RATE)
            
            total_debit = (qty_delta * price) + cost
            
            if qty_delta > 0: # Buying
                if current_pf.cash < total_debit:
                    self.logger.warning(f"MARGIN CALL: Insufficient funds for {symbol}. Req: {total_debit:.2f}, Avail: {current_pf.cash:.2f}. Order REJECTED.")
                    continue
            else: # Selling
                # [Task 019] Short Selling Constraint
                # Ensure we don't sell more than we own (Long-Only Backtest default)
                current_pos_qty = current_pf.positions[symbol].quantity
                if current_pos_qty < abs(qty_delta):
                    self.logger.warning(f"Short Sell Rejected: Insufficient holdings for {symbol}. Have: {current_pos_qty}, Sell: {abs(qty_delta)}.")
                    continue
            
            current_pf.positions[symbol].quantity += qty_delta
            current_pf.cash -= total_debit 
            
        self.pending_orders.clear()
            
    def _update_portfolio_valuation(self, market_data):
        """Update total value based on latest prices."""
        pf = self.pipeline_state.portfolio_state
        pos_value = 0.0
        
        price_map = self._extract_price_map(market_data, price_type='close')
        
        for sym, pos in pf.positions.items():
            price = price_map.get(sym, pos.average_price) 
            if price > 0:
                pos.market_value = pos.quantity * price
            pos_value += pos.market_value
            
        pf.total_value = pf.cash + pos_value

    def _extract_price_map(self, market_data, price_type='close'):
        """Helper to extract price map from batch."""
        price_map = {}
        if isinstance(market_data, dict) and "market_data" in market_data:
             for md in market_data["market_data"]:
                 price_map[md.symbol] = getattr(md, price_type)
        elif isinstance(market_data, dict):
             for sym, md in market_data.items():
                 if hasattr(md, price_type):
                     price_map[sym] = getattr(md, price_type)
        return price_map

    def get_performance_metrics(self):
        """
        Calculate real performance metrics from the equity curve.
        """
        if not self.equity_curve:
            return {}
        
        df = pd.DataFrame(self.equity_curve)
        df['returns'] = df['equity'].pct_change().fillna(0)
        
        total_return = (df['equity'].iloc[-1] - df['equity'].iloc[0]) / df['equity'].iloc[0]
        
        cumulative_max = df['equity'].cummax()
        drawdown = (df['equity'] - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()
        
        mean_ret = df['returns'].mean()
        std_ret = df['returns'].std()
        sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret != 0 else 0.0
        
        metrics = {
            "total_return": float(total_return),
            "max_drawdown": float(max_drawdown),
            "sharpe_ratio": float(sharpe),
            "final_equity": float(df['equity'].iloc[-1])
        }
        return metrics

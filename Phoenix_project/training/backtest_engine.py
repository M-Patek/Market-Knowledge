"""
Phoenix_project/training/backtest_engine.py
[Phase 4 Task 3] Refactor Backtest Matching Logic.
Implement Volume Check & Partial Fill to prevent rigid rejections.
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
    [Phase 4 Fix] Enhanced Matching: Volume Check & Partial Fills.
    """
    
    def __init__(
        self, 
        config: Dict[str, Any],
        data_manager: DataManager,
        pipeline_state: PipelineState,
        cognitive_engine: CognitiveEngine,
        portfolio_constructor: Any = None,
        order_manager: Any = None,
        risk_manager: Any = None
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
        
        self.equity_curve = []   
        self.trade_log = []      
        self.pending_orders = defaultdict(float) 

    async def run_backtest(self, data_iterator):
        self.logger.info("Starting authentic backtest run...")
        self.equity_curve = []
        self.trade_log = []
        self.pending_orders.clear()

        if not self.pipeline_state.portfolio_state:
            initial_cash = self.config.get("initial_cash", 100000.0)
            self.pipeline_state.portfolio_state = PortfolioState(
                cash=initial_cash,
                positions={},
                total_value=initial_cash
            )

        for current_time, market_data_slice in data_iterator:
            self.pipeline_state.update_time(current_time)
            setattr(self.pipeline_state, 'market_data_batch', market_data_slice)

            # [Task 4.2] Step A: Execute Pending Orders from T-1 at Current Open (T)
            await self._execute_pending_orders(market_data_slice)

            # [Task 016] Backtest Risk Integration
            if self.risk_manager:
                data_list = market_data_slice
                if hasattr(market_data_slice, 'market_data'):
                    data_list = market_data_slice.market_data
                elif isinstance(market_data_slice, dict) and 'market_data' in market_data_slice:
                    data_list = market_data_slice['market_data']
                
                if isinstance(data_list, list):
                    for item in data_list:
                        try:
                            md = item if isinstance(item, MarketData) else MarketData(**item)
                            await self.risk_manager.on_market_data(md)
                        except Exception as e:
                            self.logger.warning(f"Failed to update risk manager in backtest: {e}")

            # [Task 4.2] Step B: Generate Signals for T (to be executed at T+1)
            target_portfolio = None
            if self.portfolio_constructor:
                target_portfolio = await self.portfolio_constructor.construct_portfolio(self.pipeline_state)
                if target_portfolio:
                    self._stage_orders_for_next_bar(target_portfolio, market_data_slice)
            
            self._update_portfolio_valuation(market_data_slice)
            
            self.equity_curve.append({
                "timestamp": current_time,
                "equity": self.pipeline_state.portfolio_state.total_value,
                "cash": self.pipeline_state.portfolio_state.cash
            })

        self.logger.info("Backtest run completed.")
        return self.get_performance_metrics()

    def _stage_orders_for_next_bar(self, target_portfolio, market_data):
        current_pf = self.pipeline_state.portfolio_state
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
            
            self.pending_orders[symbol] = qty_delta

    async def _execute_pending_orders(self, market_data):
        """
        [Task 4.2] Execute pending orders at the OPEN price of the current bar (T).
        [Phase 4 Fix] Added Volume Check & Partial Fill Logic.
        """
        if not self.pending_orders:
            return
            
        current_pf = self.pipeline_state.portfolio_state
        price_map = self._extract_price_map(market_data, price_type='open')
        volume_map = self._extract_price_map(market_data, price_type='volume') 
        
        def get_volume(sym):
            return volume_map.get(sym, 0.0)

        COMMISSION_RATE = 0.001 
        SLIPPAGE_RATE = 0.0005  

        for symbol, qty_delta in list(self.pending_orders.items()):
            price = price_map.get(symbol, 0.0)
            volume = get_volume(symbol)
            
            # [Phase 4 Fix] Volume & Price Check
            if price <= 0: 
                self.logger.warning(f"Skipping {symbol}: Invalid price {price}.")
                continue
            if volume <= 0:
                self.logger.warning(f"Skipping {symbol}: Zero volume (Illiquid).")
                continue
            
            # [Task 4.2 Fix] Pre-Trade Risk Check
            if self.risk_manager:
                try:
                    proposed_trade = Position(
                        symbol=symbol,
                        quantity=qty_delta, 
                        average_price=price,
                        market_value=abs(qty_delta * price),
                        unrealized_pnl=0.0
                    )
                    self.risk_manager.check_pre_trade(proposed_trade, current_pf)
                except (RiskViolationError, CircuitBreakerError) as e:
                    self.logger.warning(f"Risk Check Failed for {symbol}: {e}. Order REJECTED.")
                    continue

            if symbol not in current_pf.positions:
                current_pf.positions[symbol] = Position(symbol=symbol, quantity=0.0, average_price=price, market_value=0.0, unrealized_pnl=0.0)
            
            # Calculate Cost
            trade_value = abs(qty_delta * price)
            cost = trade_value * (COMMISSION_RATE + SLIPPAGE_RATE)
            total_debit = trade_value + cost if qty_delta > 0 else 0
            
            actual_qty = qty_delta

            if qty_delta > 0: # Buying
                if current_pf.cash < total_debit:
                    # [Phase 4 Fix] Partial Fill Logic
                    # Cash = Q * P * (1 + rates)  => Q = Cash / (P * (1 + rates))
                    max_qty = current_pf.cash / (price * (1 + COMMISSION_RATE + SLIPPAGE_RATE))
                    
                    if max_qty < 0.0001: # Dust check
                        self.logger.warning(f"REJECTED {symbol}: Insufficient funds even for partial fill.")
                        continue
                        
                    self.logger.info(f"PARTIAL FILL {symbol}: Requested {qty_delta}, Afford {max_qty:.4f}.")
                    actual_qty = max_qty
                    
                    # Recalculate cost for actual qty
                    trade_value = actual_qty * price
                    cost = trade_value * (COMMISSION_RATE + SLIPPAGE_RATE)
                    total_debit = trade_value + cost

            else: # Selling
                # [Task 019] Short Selling Constraint
                current_pos_qty = current_pf.positions[symbol].quantity
                if current_pos_qty < abs(qty_delta):
                    self.logger.warning(f"Short Sell Rejected: Insufficient holdings for {symbol}. Have: {current_pos_qty}, Sell: {abs(qty_delta)}.")
                    continue
                # Credit cash (Sell value - cost)
                credit = abs(qty_delta * price) - cost
                current_pf.cash += credit
                total_debit = 0 # Already handled

            if qty_delta > 0:
                current_pf.cash -= total_debit
            
            current_pf.positions[symbol].quantity += actual_qty
            
            self.trade_log.append({
                "symbol": symbol,
                "quantity": actual_qty,
                "price": price,
                "cost": cost,
                "type": "BUY" if actual_qty > 0 else "SELL"
            })
            
        self.pending_orders.clear()
            
    def _update_portfolio_valuation(self, market_data):
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
        price_map = {}
        if isinstance(market_data, dict) and "market_data" in market_data:
             for md in market_data["market_data"]:
                 val = getattr(md, price_type, None)
                 if val is not None: price_map[md.symbol] = float(val)
        elif isinstance(market_data, dict):
             for sym, md in market_data.items():
                 if hasattr(md, price_type):
                     val = getattr(md, price_type, None)
                     if val is not None: price_map[sym] = float(val)
                 elif isinstance(md, dict):
                     val = md.get(price_type)
                     if val is not None: price_map[sym] = float(val)
        return price_map

    def get_performance_metrics(self):
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

"""
Phoenix_project/training/backtest_engine.py
[Phase 4 Task 3] Refactor Backtest Matching Logic.
Implement Volume Check & Partial Fill to prevent rigid rejections.
[Task P1-001] Integrate SimulatedBroker for consistent execution logic with RL Env.
"""
from Phoenix_project.monitor.logging import get_logger
from Phoenix_project.data_manager import DataManager
from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.core.schemas.data_schema import PortfolioState, Position, MarketData, Order, OrderType, OrderStatus
from Phoenix_project.cognitive.engine import CognitiveEngine
from Phoenix_project.core.exceptions import RiskViolationError, CircuitBreakerError
from Phoenix_project.execution.adapters import SimulatedBroker
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from collections import defaultdict
from decimal import Decimal
import asyncio

class BacktestEngine:
    """
    用于 Walk-Forward 训练和评估的模拟引擎。
    [Phase II Fix] 集成 RiskManager 进行预交易检查，防止过拟合。
    [Phase 4 Fix] Enhanced Matching: Volume Check & Partial Fills.
    [Task P1-001] Consistent Execution via SimulatedBroker.
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
        
        # [Task P1-001] Initialize Simulated Broker
        broker_config = self.config.copy()
        broker_config['initial_cash'] = self.config.get("initial_cash", 100000.0)
        self.broker = SimulatedBroker(broker_config)
        self.broker.connect()
        
        self.logger.info("BacktestEngine initialized with SimulatedBroker.")
        
        self.equity_curve = []   
        self.trade_log = []      
        self.pending_orders = defaultdict(float) 

    async def run_backtest(self, data_iterator):
        self.logger.info("Starting authentic backtest run...")
        self.equity_curve = []
        self.trade_log = []
        self.pending_orders.clear()
        
        # Reset Broker state if needed (SimulatedBroker doesn't have reset, so we rely on fresh instance or re-init if looped)
        # For now, we assume one run per instance or handled externally.

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
            # Now delegates to self.broker
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
        """
        Calculates target quantities based on Close prices of T.
        These become pending orders for T+1.
        """
        # Use PipelineState which is synced with Broker
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
        [Task P1-001] Uses SimulatedBroker for state management.
        """
        if not self.pending_orders:
            return
            
        # [Task P1-001] Execution Logic Alignment
        # Match RL Env constants
        COMMISSION_RATE = 0.001 
        
        price_map = self._extract_price_map(market_data, price_type='open')
        volume_map = self._extract_price_map(market_data, price_type='volume') 
        spread_map = self._extract_price_map(market_data, price_type='spread') # If available
        
        current_pf = self.pipeline_state.portfolio_state

        for symbol, qty_delta in list(self.pending_orders.items()):
            price = price_map.get(symbol, 0.0)
            volume = volume_map.get(symbol, 0.0)
            spread = spread_map.get(symbol, 0.001)
            
            if price <= 0: 
                self.logger.warning(f"Skipping {symbol}: Invalid price {price}.")
                continue
            
            # [Phase 4 Fix] Volume Check
            if volume <= 0:
                self.logger.warning(f"Skipping {symbol}: Zero volume (Illiquid).")
                continue
            
            # [Task P1-001] Consistent Slippage Model with TradingEnv
            # slippage_rate = (spread / 2.0) + 0.0005
            slippage_rate = (spread / 2.0) + 0.0005
            
            # Calculate execution price impact
            # Buy executes higher, Sell executes lower
            side_sign = 1 if qty_delta > 0 else -1
            exec_price = price * (1 + slippage_rate * side_sign)
            
            # [Task 4.2 Fix] Pre-Trade Risk Check
            if self.risk_manager:
                try:
                    proposed_trade = Position(
                        symbol=symbol,
                        quantity=qty_delta, 
                        average_price=exec_price,
                        market_value=abs(qty_delta * exec_price),
                        unrealized_pnl=0.0
                    )
                    self.risk_manager.check_pre_trade(proposed_trade, current_pf)
                except (RiskViolationError, CircuitBreakerError) as e:
                    self.logger.warning(f"Risk Check Failed for {symbol}: {e}. Order REJECTED.")
                    continue

            # [Phase 4 Fix] Partial Fill Logic based on Broker Cash
            broker_cash = float(self.broker.get_cash_balance())
            actual_qty = qty_delta

            if qty_delta > 0: # Buying
                # Cost estimate: Q * P * (1 + Comm)
                # Note: SimulatedBroker deducts Q*P. We assume Comm is extra.
                estimated_total_cost = actual_qty * exec_price * (1 + COMMISSION_RATE)
                
                if broker_cash < estimated_total_cost:
                     # Downgrade quantity
                     max_qty = broker_cash / (exec_price * (1 + COMMISSION_RATE))
                     if max_qty < 0.0001:
                         self.logger.warning(f"REJECTED {symbol}: Insufficient funds ({broker_cash}) for trade.")
                         continue
                     actual_qty = max_qty
                     self.logger.info(f"PARTIAL FILL {symbol}: Requested {qty_delta:.4f}, Adjusted to {actual_qty:.4f}")

            # [Task P1-001] Execute via Broker
            try:
                order = Order(
                    id="", 
                    symbol=symbol,
                    quantity=Decimal(str(actual_qty)),
                    order_type=OrderType.MARKET,
                    status=OrderStatus.NEW
                )
                
                # Execute
                self.broker.place_order(order, price=exec_price)
                
                # Manual Fee Deduction (Alignment with TradingEnv)
                trade_value = abs(actual_qty * exec_price)
                fee = trade_value * COMMISSION_RATE
                self.broker.cash -= Decimal(str(fee))
                
                # Log
                slippage_cost = abs(trade_value * slippage_rate)
                self.trade_log.append({
                    "symbol": symbol,
                    "quantity": actual_qty,
                    "price": exec_price,
                    "cost": fee + slippage_cost,
                    "type": "BUY" if actual_qty > 0 else "SELL"
                })
                
            except Exception as e:
                self.logger.error(f"Broker Execution Failed for {symbol}: {e}")

        self.pending_orders.clear()
        
        # [Task P1-001] Sync PipelineState with Broker
        await self._sync_pipeline_state()

    async def _sync_pipeline_state(self):
        """
        Synchronizes the internal pipeline_state.portfolio_state with the SimulatedBroker's authoritative state.
        """
        pf_info = self.broker.get_account_info()
        self.pipeline_state.portfolio_state.cash = float(pf_info['account']['cash'])
        
        # Rebuild positions dict
        new_positions = {}
        for p in pf_info['positions']:
            sym = p['symbol']
            new_positions[sym] = Position(
                symbol=sym,
                quantity=float(p['qty']),
                average_price=float(p.get('avg_entry_price', 0.0)),
                market_value=float(p.get('market_value', 0.0)),
                unrealized_pnl=float(p.get('unrealized_pl', 0.0))
            )
        self.pipeline_state.portfolio_state.positions = new_positions

    def _update_portfolio_valuation(self, market_data):
        """
        Updates the total_value of the portfolio based on latest Close prices.
        """
        # First sync quantities/cash from broker
        # (Already done in _execute_pending_orders but good to ensure if logic changes)
        # self._sync_pipeline_state() # Removed to avoid double async call issues if not awaited properly in sync context
        
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

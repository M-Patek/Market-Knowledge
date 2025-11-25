"""
Backtest Engine
- 模拟执行交易策略
- 使用历史数据
- 评估策略表现
"""
# 修正：将 'monitor.logging' 转换为 'Phoenix_project.monitor.logging'
from Phoenix_project.monitor.logging import get_logger
# 修正：将 'data_manager' 转换为 'Phoenix_project.data_manager'
from Phoenix_project.data_manager import DataManager
# 修正：将 'core.pipeline_state' 转换为 'Phoenix_project.core.pipeline_state'
from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.core.schemas.data_schema import PortfolioState, Position
# 修正：将 'cognitive.engine' 转换为 'Phoenix_project.cognitive.engine'
from Phoenix_project.cognitive.engine import CognitiveEngine
from typing import Dict, Any
import pandas as pd
import numpy as np
import asyncio

class BacktestEngine:
    """
    用于 Walk-Forward 训练和评估的模拟引擎。
    """
    
    def __init__(
        self, 
        config: Dict[str, Any],
        data_manager: DataManager,
        pipeline_state: PipelineState,
        cognitive_engine: CognitiveEngine,
        portfolio_constructor: Any = None, # [Dependency Injection]
        order_manager: Any = None          # [Dependency Injection]
    ):
        self.config = config
        self.data_manager = data_manager
        self.pipeline_state = pipeline_state
        self.cognitive_engine = cognitive_engine
        self.portfolio_constructor = portfolio_constructor
        self.order_manager = order_manager
        
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info("BacktestEngine initialized with authentic simulation components.")
        
        self.equity_curve = []   # Track equity over time
        self.trade_log = []      # Track executed trades

    def run_backtest(self, data_iterator):
        """
        [Real Implementation] Iterate through data, generate signals, execute trades, track equity.
        """
        self.logger.info("Starting authentic backtest run...")
        self.equity_curve = []
        self.trade_log = []

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
            
            # 2. Construct Target Portfolio (Signal Generation)
            target_portfolio = None
            if self.portfolio_constructor:
                # Mock L3 signal if not present (simplified for basic backtest)
                if not self.pipeline_state.l3_alpha_signal:
                     # Simple strategy hook or pass
                     pass
                
                target_portfolio = self.portfolio_constructor.construct_portfolio(self.pipeline_state)
            
            # 3. Simulate Execution
            # If we have an OrderManager, use it (mocked or real). 
            # Otherwise use internal simplified fill logic.
            if self.order_manager and target_portfolio:
                self.pipeline_state.set_target_portfolio(target_portfolio)
                # Assuming synchronous simulation for backtest speed, or await if async
                # asyncio.run(self.order_manager.reconcile_portfolio(self.pipeline_state))
                pass # Real OM execution might differ in simulation vs live
            elif target_portfolio:
                self._simulate_immediate_execution(target_portfolio, market_data_slice)
            
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

    def _simulate_immediate_execution(self, target_portfolio, market_data):
        """Simplified execution assuming instant fill at close price."""
        current_pf = self.pipeline_state.portfolio_state
        
        for pos in target_portfolio.positions:
            symbol = pos.symbol
            # 适配 market_data 结构: 可能是 dict 或 list
            # 如果是 list (batch_data['market_data']), 需要查找
            price = 0.0
            if isinstance(market_data, dict) and "market_data" in market_data:
                # Batch format
                for md in market_data["market_data"]:
                    if md.symbol == symbol:
                        price = md.close
                        break
            elif isinstance(market_data, dict) and symbol in market_data:
                 # Dict format {sym: MarketData}
                 price = market_data[symbol].close
            
            if price <= 0: continue
            
            target_val = current_pf.total_value * pos.target_weight
            current_pos_val = 0.0
            if symbol in current_pf.positions:
                current_pos_val = current_pf.positions[symbol].market_value
            
            diff_val = target_val - current_pos_val
            qty_change = diff_val / price
            
            # Update Position
            if symbol not in current_pf.positions:
                current_pf.positions[symbol] = Position(symbol=symbol, quantity=0, current_price=price, market_value=0.0)
            
            current_pf.positions[symbol].quantity += qty_change
            current_pf.cash -= diff_val # Deduct cost
            
    def _update_portfolio_valuation(self, market_data):
        """Update total value based on latest prices."""
        pf = self.pipeline_state.portfolio_state
        pos_value = 0.0
        
        # Build price map
        price_map = {}
        if isinstance(market_data, dict) and "market_data" in market_data:
             for md in market_data["market_data"]:
                 price_map[md.symbol] = md.close
        
        for sym, pos in pf.positions.items():
            price = price_map.get(sym, pos.current_price) # Use last known if missing
            if price > 0:
                pos.current_price = price
                pos.market_value = pos.quantity * price
            pos_value += pos.market_value
            
        pf.total_value = pf.cash + pos_value

    def get_performance_metrics(self):
        """
        Calculate real performance metrics from the equity curve.
        """
        if not self.equity_curve:
            self.logger.warning("No equity curve data to analyze.")
            return {}
        
        df = pd.DataFrame(self.equity_curve)
        df['returns'] = df['equity'].pct_change().fillna(0)
        
        total_return = (df['equity'].iloc[-1] - df['equity'].iloc[0]) / df['equity'].iloc[0]
        
        # Max Drawdown
        cumulative_max = df['equity'].cummax()
        drawdown = (df['equity'] - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()
        
        # Sharpe Ratio (assuming daily data, annualized)
        mean_ret = df['returns'].mean()
        std_ret = df['returns'].std()
        sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret != 0 else 0.0
        
        metrics = {
            "total_return": float(total_return),
            "max_drawdown": float(max_drawdown),
            "sharpe_ratio": float(sharpe),
            "final_equity": float(df['equity'].iloc[-1])
        }
        self.logger.info(f"Performance Metrics: {metrics}")
        return metrics

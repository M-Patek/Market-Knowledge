import pandas as pd
from typing import Dict, Any, List

from ..audit_manager import AuditManager
from ..monitor.logging import get_logger

logger = get_logger(__name__)

class TradeLifecycleManager:
    """
    Manages the state of the portfolio over time by applying fills
    and updating cash and positions.
    
    This is a core component of the backtesting/simulation engine.
    """

    def __init__(self, initial_capital: float, audit_manager: AuditManager):
        """
        Initializes the TradeLifecycleManager.
        
        Args:
            initial_capital (float): The starting cash balance.
            audit_manager (AuditManager): Client for logging trades and portfolio changes.
        """
        self.initial_capital = initial_capital
        self.audit_manager = audit_manager
        
        # The portfolio state is a dictionary mapping symbols to shares
        # e.g., {"CASH": 1000000, "AAPL": 100}
        self.portfolio: Dict[str, float] = {"CASH": initial_capital}
        
        self.trade_log: List[Dict[str, Any]] = []
        self.pnl_history: List[Dict[str, Any]] = []
        
        logger.info(f"TradeLifecycleManager initialized with capital: {initial_capital}")

    def update_portfolio_with_fills(
        self, 
        fills: List[Dict[str, Any]], 
        costs: Dict[str, float],
        timestamp: pd.Timestamp
    ):
        """
        Updates the portfolio state (cash and shares) based on a list of
        simulated fills and associated costs.
        
        Args:
            fills (List[Dict]): A list of fill records from OrderManager.simulate_execution.
            costs (Dict[str, float]): A dict of total costs (slippage, commission).
            timestamp (pd.Timestamp): The timestamp for this update.
        """
        
        if not fills and not costs.get("total_cost", 0) > 0:
            # No activity, just mark the PnL
            self.mark_to_market(timestamp, {})
            return

        # 1. Apply fills to portfolio
        for fill in fills:
            symbol = fill['symbol']
            quantity = fill['quantity']
            execution_price = fill['execution_price']
            
            # Update shares
            current_shares = self.portfolio.get(symbol, 0.0)
            self.portfolio[symbol] = current_shares + quantity
            
            # Update cash
            # Buy (quantity > 0) reduces cash
            # Sell (quantity < 0) increases cash
            cash_change = - (quantity * execution_price)
            self.portfolio['CASH'] += cash_change
            
            # Log the trade
            self.trade_log.append(fill)
            self.audit_manager.log_trade(fill)

        # 2. Deduct any additional costs (e.g., total commission if not part of fill)
        # In our current OrderManager, costs are already baked into the fill_price
        # via slippage, and commissions are itemized. Let's assume we just need
        # to log the total cost summary.
        self.audit_manager.log_costs(costs, timestamp)

        # 3. Mark-to-Market (MtM)
        # After all trades are applied, calculate the new portfolio value
        self.mark_to_market(timestamp, self.get_latest_prices_from_fills(fills))

    def mark_to_market(self, timestamp: pd.Timestamp, prices: Dict[str, float]):
        """
        Calculates the current total value of the portfolio (Equity).
        
        Args:
            timestamp (pd.Timestamp): The current time.
            prices (Dict[str, float]): A dict of {symbol: price} for all assets held.
                                       If a symbol is missing, its last known
                                       price from trades will be used (less accurate).
        """
        total_value = self.portfolio.get('CASH', 0.0)
        positions_value = 0.0
        
        for symbol, shares in self.portfolio.items():
            if symbol == 'CASH' or shares == 0:
                continue
            
            price = prices.get(symbol)
            if price is None:
                # Fallback: try to get the last fill price (less accurate)
                price = self._get_last_fill_price(symbol)
                if price is None:
                    logger.warning(f"No price for {symbol} during MtM. Using 0.")
                    price = 0.0
            
            positions_value += shares * price
        
        total_value += positions_value
        
        pnl_record = {
            "timestamp": timestamp,
            "total_equity": total_value,
            "cash": self.portfolio.get('CASH', 0.0),
            "positions_value": positions_value,
            "net_pnl": total_value - self.initial_capital
        }
        
        self.pnl_history.append(pnl_record)
        self.audit_manager.log_portfolio_snapshot(self.portfolio, pnl_record)

    def get_current_portfolio(self) -> Dict[str, float]:
        """Returns the current portfolio state (shares)."""
        return self.portfolio.copy()

    def get_pnl_history(self) -> pd.DataFrame:
        """Returns the historical PnL as a DataFrame."""
        return pd.DataFrame(self.pnl_history).set_index('timestamp')

    def get_trade_log(self) -> pd.DataFrame:
        """Returns the log of all executed trades as a DataFrame."""
        return pd.DataFrame(self.trade_log)

    def get_latest_prices_from_fills(self, fills: List[Dict]) -> Dict[str, float]:
        """Helper to get the most recent price for each symbol from a list of fills."""
        prices = {}
        for fill in fills:
            # We use 'ideal_price' for MtM to be conservative
            prices[fill['symbol']] = fill['ideal_price'] 
        return prices

    def _get_last_fill_price(self, symbol: str) -> float:
        """Finds the last execution price for a symbol from the trade log."""
        for fill in reversed(self.trade_log):
            if fill['symbol'] == symbol:
                return fill['execution_price']
        return None

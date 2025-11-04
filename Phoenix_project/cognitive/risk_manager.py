from typing import List, Dict, Any, Optional
from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.core.schemas.data_schema import Order, Signal
from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

class RiskManager:
    """
    Provides pre-trade and post-trade risk checks.
    It can veto orders or signals based on predefined rules.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("risk_manager", {})
        
        # Pre-trade limits
        self.max_order_value = self.config.get("max_order_value", 20000) # Max $20k per order
        self.max_order_quantity = self.config.get("max_order_quantity", 1000) # Max 1000 shares
        
        # Portfolio limits
        self.max_portfolio_drawdown = self.config.get("max_portfolio_drawdown", 0.15) # 15%
        self.max_position_concentration = self.config.get("max_position_concentration", 0.20) # 20%
        
        # Circuit breaker
        self.circuit_breaker_tripped = False
        self.circuit_breaker_reason = ""
        
        logger.info("RiskManager initialized.")

    def trip_circuit_breaker(self, reason: str):
        """Trips the system-wide circuit breaker, halting all new trades."""
        if not self.circuit_breaker_tripped:
            logger.critical(f"CIRCUIT BREAKER TRIPPED: {reason}")
            self.circuit_breaker_tripped = True
            self.circuit_breaker_reason = reason
            # This should also publish an event
            # await pipeline_state.event_distributor.publish("CIRCUIT_BREAKER", reason=reason)

    def reset_circuit_breaker(self):
        """Resets the circuit breaker (manual action)."""
        logger.warning("Circuit breaker is being reset.")
        self.circuit_breaker_tripped = False
        self.circuit_breaker_reason = ""

    def check_portfolio_risk(self, pipeline_state: PipelineState):
        """
        Checks overall portfolio health (e.g., drawdown).
        This can trip the circuit breaker.
        """
        if self.circuit_breaker_tripped:
            return # Already tripped

        portfolio_metrics = pipeline_state.get_value("portfolio_metrics", {})
        drawdown = portfolio_metrics.get("current_drawdown", 0.0)
        
        if drawdown > self.max_portfolio_drawdown:
            reason = f"Maximum portfolio drawdown exceeded ({drawdown:.2%} > {self.max_portfolio_drawdown:.2%})"
            self.trip_circuit_breaker(reason)

    async def validate_signal(self, signal: Signal, pipeline_state: PipelineState) -> Optional[str]:
        """
        Checks a signal *before* it's turned into an order.
        Returns a rejection reason string if invalid, or None if valid.
        """
        if self.circuit_breaker_tripped:
            return f"Circuit breaker tripped: {self.circuit_breaker_reason}"
            
        if abs(signal.direction) > 1:
            return "Invalid signal direction."
            
        if not (0.0 <= signal.strength <= 1.0):
            return f"Invalid signal strength: {signal.strength}"
            
        # Add more signal-level checks (e.g., duplicate signals)
        
        return None # Signal is valid

    async def validate_order(self, order: Order, pipeline_state: PipelineState) -> Optional[str]:
        """
        Performs pre-trade checks on a generated order.
        Returns a rejection reason string if invalid, or None if valid.
        """
        if self.circuit_breaker_tripped:
            return f"Circuit breaker tripped: {self.circuit_breaker_reason}"

        # Get context
        market_data = pipeline_state.get_latest_market_data(order.symbol)
        if not market_data:
            return "No market data available to price order."
            
        current_price = market_data.close
        order_value = abs(order.quantity * current_price)
        
        # 1. Check order-level limits
        if order_value > self.max_order_value:
            return f"Order value ${order_value:.2f} exceeds max limit ${self.max_order_value:.2f}"
            
        if abs(order.quantity) > self.max_order_quantity:
            return f"Order quantity {order.quantity} exceeds max limit {self.max_order_quantity}"
            
        # 2. Check portfolio-level impact (concentration)
        portfolio = pipeline_state.get_value("portfolio", {})
        total_value = portfolio.get("total_value", 0)
        if total_value == 0:
            return "Portfolio value is zero."
            
        positions = portfolio.get("positions", {})
        current_position = positions.get(order.symbol, {})
        current_value = current_position.get("market_value", 0)
        
        # Calculate post-trade value
        post_trade_value = current_value + (order.quantity * current_price)
        
        # Calculate post-trade concentration
        post_trade_concentration = abs(post_trade_value) / total_value
        
        if post_trade_concentration > self.max_position_concentration:
            return f"Order would increase position concentration to {post_trade_concentration:.2%}, exceeding limit {self.max_position_concentration:.2%}"

        # Add more checks (e.g., buying power)
        
        return None # Order is valid

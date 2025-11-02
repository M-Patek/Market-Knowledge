from typing import Dict, List, Any
from core.pipeline_state import PipelineState
from core.schemas.data_schema import Signal, Order
from core.schemas.fusion_result import AgentDecision # This will fail
from sizing.base import SizingMethod
from monitor.logging import get_logger

logger = get_logger(__name__)

class PortfolioConstructor:
    """
    Generates target portfolio weights and translates cognitive decisions
    (Signals) into concrete Orders.
    
    This module bridges the "AI brain" (CognitiveEngine) and the
    "execution arm" (OrderManager).
    """

    def __init__(self, sizing_method: SizingMethod, config: Dict[str, Any]):
        self.sizing_method = sizing_method
        self.config = config
        self.max_position_size = config.get("max_position_size", 0.1) # Max 10% of portfolio
        self.min_order_value = config.get("min_order_value", 100) # Min $100 per order
        logger.info(f"PortfolioConstructor initialized with sizing method: {sizing_method.__class__.__name__}")

    def generate_signal(self, fusion_result: Any) -> Signal: # FusionResult
        """
        Converts the final decision from the CognitiveEngine into a
        standardized Signal object.
        
        Args:
            fusion_result (FusionResult): The output from the cognitive cycle.

        Returns:
            Signal: A standardized signal object.
        """
        
        # Map decision to signal direction
        decision = fusion_result.final_decision.upper()
        if decision == "BUY":
            direction = 1
        elif decision == "SELL":
            direction = -1
        else: # HOLD, ERROR_HOLD, etc.
            direction = 0
            
        # Use confidence as the "weight" or "strength" of the signal
        strength = fusion_result.confidence
        
        # TODO: Get symbol from config or state
        symbol = "PRIMARY_ASSET" 
        
        signal = Signal(
            symbol=symbol,
            timestamp=datetime.utcnow(),
            direction=direction,
            strength=strength,
            signal_type="AI_COGNITIVE",
            metadata={
                "reasoning": fusion_result.reasoning[:500], # Truncate
                "decision_id": fusion_result.decision_id # Assume ID is attached
            }
        )
        logger.info(f"Generated signal: {signal.direction} @ {signal.strength:.2f} for {signal.symbol}")
        return signal

    def generate_orders_from_signal(
        self,
        signal: Signal,
        pipeline_state: PipelineState
    ) -> List[Order]:
        """
        Calculates the required trade size based on the signal and
        current portfolio state, then generates Order objects.
        
        Args:
            signal (Signal): The signal from `generate_signal`.
            pipeline_state (PipelineState): The current system state.
            
        Returns:
            List[Order]: A list of orders to be executed (can be empty).
        """
        
        current_portfolio = pipeline_state.get_value("portfolio", {})
        current_positions = current_portfolio.get("positions", {})
        total_value = current_portfolio.get("total_value", 100000) # Default portfolio val
        
        current_position_size = current_positions.get(signal.symbol, {}).get("size", 0)
        
        market_data = pipeline_state.get_latest_market_data(signal.symbol)
        if not market_data:
            logger.error(f"Cannot generate order for {signal.symbol}: No market data.")
            return []
            
        current_price = market_data.close

        # Use the sizing method to determine the *target* position size
        target_size_dollars = self.sizing_method.calculate_target_size(
            signal=signal,
            current_price=current_price,
            portfolio_value=total_value,
            current_position_size=current_position_size
        )
        
        # Apply portfolio constraints
        max_size_dollars = total_value * self.max_position_size
        if target_size_dollars > max_size_dollars:
            logger.warning(f"Sizing method target ({target_size_dollars}) exceeds max position size. Capping at {max_size_dollars}.")
            target_size_dollars = max_size_dollars
        elif target_size_dollars < -max_size_dollars:
            logger.warning(f"Sizing method target ({target_size_dollars}) exceeds max position size. Capping at {-max_size_dollars}.")
            target_size_dollars = -max_size_dollars
            
        # Calculate target quantity (shares)
        target_quantity = target_size_dollars / current_price
        
        # Calculate the delta (how much to trade)
        current_quantity = current_position_size # Assume this is in shares
        order_quantity = target_quantity - current_quantity
        
        # Apply minimum order constraints
        order_value = abs(order_quantity * current_price)
        if order_value < self.min_order_value:
            logger.info(f"Calculated order value ${order_value:.2f} is below minimum ${self.min_order_value}. No order generated.")
            return []
            
        # Create the order
        if order_quantity == 0:
            return []
            
        order_type = "MARKET" # Or "LIMIT" etc.
        
        order = Order(
            symbol=signal.symbol,
            timestamp=datetime.utcnow(),
            quantity=order_quantity,
            order_type=order_type,
            # price=... (if limit order)
            status="PENDING",
            signal_id=signal.id # Link order to signal
        )
        
        logger.info(f"Generated order: {order.order_type} {order.quantity:.4f} {order.symbol}")
        return [order]

from typing import Dict, Any, List
from core.schemas.data_schema import Signal
from monitor.logging import get_logger

logger = get_logger(__name__)

class SignalProtocol:
    """
    Defines and validates the structure of signals used within
    the system.
    
    This class is mostly a validator and standardizer. The main
    schema is defined in `core.schemas.data_schema.Signal`.
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.known_signal_types = self.config.get("known_signal_types", [
            "AI_COGNITIVE",
            "TECHNICAL_ANALYSIS",
            "MANUAL_OVERRIDE"
        ])
        logger.info("SignalProtocol initialized.")

    def validate_signal(self, signal_data: Dict[str, Any]) -> (bool, str, Signal):
        """
        Validates raw signal data and attempts to parse it into
        a standardized Signal object.
        
        Returns:
            (bool, str, Signal): (is_valid, error_message, parsed_signal)
        """
        try:
            signal = Signal(**signal_data)
            
            # Additional semantic validation
            if signal.signal_type not in self.known_signal_types:
                return (False, f"Unknown signal_type: {signal.signal_type}", None)
                
            if not (-1 <= signal.direction <= 1):
                return (False, f"Invalid direction: {signal.direction}", None)
                
            if not (0.0 <= signal.strength <= 1.0):
                 return (False, f"Invalid strength: {signal.strength}", None)
                 
            return (True, "", signal)
            
        except Exception as e: # e.g., Pydantic ValidationError
            logger.warning(f"Failed to validate signal: {e}")
            return (False, f"Validation error: {e}", None)
            
    def create_signal(self, *args, **kwargs) -> Signal:
        """Helper method to create a valid Signal object."""
        # This just passes through to the Pydantic model
        return Signal(*args, **kwargs)

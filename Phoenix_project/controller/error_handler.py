import traceback
from typing import Optional

from ..monitor.logging import get_logger

logger = get_logger(__name__)

class CircuitBreaker:
    """
    A simple circuit breaker mechanism to stop the system
    if too many consecutive errors occur.
    """
    def __init__(self, failure_threshold: int, recovery_timeout: int):
        """
        Initializes the circuit breaker.
        
        Args:
            failure_threshold (int): Number of consecutive failures to trip.
            recovery_timeout (int): Seconds to wait in 'OPEN' state before
                                    moving to 'HALF_OPEN'.
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.state = "CLOSED" # Can be CLOSED, OPEN, HALF_OPEN
        self.last_failure_time = None
        
        logger.info(f"CircuitBreaker initialized: Threshold={failure_threshold}, Timeout={recovery_timeout}s")

    def is_open(self) -> bool:
        """
        Checks if the circuit is 'OPEN'. If it is, it checks if
        the recovery timeout has passed to move to 'HALF_OPEN'.
        
        Returns:
            bool: True if calls should be blocked (circuit is OPEN).
        """
        if self.state == "OPEN":
            import time
            if (time.time() - self.last_failure_time) > self.recovery_timeout:
                self.state = "HALF_OPEN"
                logger.warning("CircuitBreaker moving to HALF_OPEN state.")
                return False # Allow one call through
            else:
                return True # Still in timeout, block call
        
        return False # Circuit is CLOSED or HALF_OPEN

    def record_failure(self):
        """Records a failure. Trips the circuit if threshold is met."""
        self.failure_count += 1
        if self.state == "HALF_OPEN":
            # Failure in HALF_OPEN state, trip back to OPEN
            self._trip()
        elif self.failure_count >= self.failure_threshold:
            self._trip()
            
    def record_success(self):
        """Records a success. Resets the circuit if applicable."""
        if self.state == "HALF_OPEN":
            self.reset()
            logger.info("CircuitBreaker reset to CLOSED state after HALF_OPEN success.")
        elif self.failure_count > 0:
            self.failure_count = 0
            # logger.debug("CircuitBreaker failure count reset.")

    def reset(self):
        """Resets the circuit to the 'CLOSED' state."""
        self.state = "CLOSED"
        self.failure_count = 0
        self.last_failure_time = None

    def _trip(self):
        """Trips the circuit to the 'OPEN' state."""
        import time
        self.state = "OPEN"
        self.last_failure_time = time.time()
        self.failure_count = 0 # Reset count after tripping
        logger.error(f"CircuitBreaker TRIPPED! Moving to OPEN state for {self.recovery_timeout}s.")


class ErrorHandler:
    """
    Centralized error handling component.
    Includes a circuit breaker for critical, repeating failures.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the ErrorHandler.
        
        Args:
            config (Dict, Any): The main system configuration.
        """
        self.config = config.get('error_handler', {})
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.get('circuit_breaker_threshold', 5),
            recovery_timeout=self.config.get('circuit_breaker_timeout_sec', 300)
        )
        
        # This logger is for the ErrorHandler itself
        self.logger = logger 

    def handle_error(
        self, 
        error: Exception, 
        context: str, 
        is_critical: bool = False,
        audit_manager: Optional[Any] = None
    ):
        """
        Main error handling method.
        
        Args:
            error (Exception): The exception that occurred.
            context (str): A string describing where the error happened (e.g., "StreamProcessor").
            is_critical (bool): If True, this failure counts against the circuit breaker.
            audit_manager (Optional[AuditManager]): To log the error to the audit trail.
        """
        
        error_message = f"Error in {context}: {type(error).__name__} - {error}"
        self.logger.error(error_message, exc_info=True)
        
        # 1. Log to audit trail
        if audit_manager:
            try:
                audit_manager.log_system_error(error, context)
            except Exception as audit_e:
                self.logger.critical(f"Failed to log error to audit trail: {audit_e}")
        
        # 2. Update Circuit Breaker
        if is_critical:
            self.logger.warning(f"Recording CRITICAL failure for circuit breaker. Context: {context}")
            self.circuit_breaker.record_failure()
            
            if self.circuit_breaker.is_open():
                self.logger.critical("Circuit breaker is OPEN. System operations may be paused.")
                # TODO: Add logic to actually *stop* the system (e.g., signal LoopManager)
                self.notify_admin(f"CRITICAL: Circuit Breaker TRIPPED due to error in {context}")

    def notify_admin(self, message: str, subject: str = "Phoenix System Alert"):
        """
        Placeholder for sending an alert to an administrator (e.g., email, PagerDuty).
        
        Args:
            message (str): The alert message.
            subject (str): The alert subject.
        """
        self.logger.critical(f"ADMIN_NOTIFICATION (Placeholder): {subject} - {message}")
        # TODO: Implement actual notification logic (e.g., SMTP, OpsGenie API)

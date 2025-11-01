from typing import Dict, Any

from ..monitor.logging import get_logger
from ..core.schemas.fusion_result import FusionResult

logger = get_logger(__name__)

class UncertaintyGuard:
    """
    A "circuit breaker" that sits *after* the cognitive pipeline.
    
    It inspects the final FusionResult. If the cognitive uncertainty
    is too high, or if the AI's decision is "INVALID" or "ERROR",
    this guard can halt the execution pipeline (e.g., prevent a
    trade) and flag the event for human review.
    """

    def __init__(self, config: Dict[str, Any], error_handler: Any):
        """
        Initializes the UncertaintyGuard.
        
        Args:
            config: Main system configuration.
            error_handler: The main ErrorHandler to notify admins.
        """
        self.config = config.get('uncertainty_guard', {})
        self.error_handler = error_handler
        
        # The uncertainty score (0.0 to 1.0) above which we block execution
        self.uncertainty_threshold = self.config.get('uncertainty_threshold', 0.75)
        
        logger.info(f"UncertaintyGuard initialized with threshold: {self.uncertainty_threshold:.0%}")

    def validate_result(self, fusion_result: FusionResult) -> bool:
        """
        Validates the FusionResult.
        
        Args:
            fusion_result (FusionResult): The output from the Synthesizer.
            
        Returns:
            bool: True if the result is "safe" to execute, False otherwise.
        """
        
        # 1. Check for pipeline errors
        if fusion_result.status == "ERROR":
            logger.error(f"UncertaintyGuard BLOCKED (PIPELINE_ERROR) event: {fusion_result.event_id}. "
                         f"Error: {fusion_result.error_message}")
            self.error_handler.notify_admin(
                f"Execution blocked for {fusion_result.event_id}: Pipeline Error - {fusion_result.error_message}"
            )
            return False
            
        # 2. Check for invalid final decisions
        if fusion_result.final_decision.decision in ["ERROR", "INVALID_RESPONSE", "UNKNOWN"]:
            logger.error(f"UncertaintyGuard BLOCKED (INVALID_DECISION) event: {fusion_result.event_id}. "
                         f"Decision: {fusion_result.final_decision.decision}")
            self.error_handler.notify_admin(
                f"Execution blocked for {fusion_result.event_id}: Invalid AI Decision - {fusion_result.final_decision.justification}"
            )
            return False

        # 3. Check uncertainty threshold
        if fusion_result.cognitive_uncertainty > self.uncertainty_threshold:
            logger.warning(f"UncertaintyGuard BLOCKED (HIGH_UNCERTAINTY) event: {fusion_result.event_id}. "
                           f"Score: {fusion_result.cognitive_uncertainty:.2f} > {self.uncertainty_threshold:.2f}")
            
            # This is a *warning*, not a critical error, but we still notify
            self.error_handler.notify_admin(
                f"Execution blocked for {fusion_result.event_id}: High Cognitive Uncertainty ({fusion_result.cognitive_uncertainty:.0%})"
            )
            return False
            
        # 4. Check for "HOLD"
        # We don't block "HOLD", as "HOLD" is a valid signal (e.g., to close positions)
        if fusion_result.final_decision.decision == "HOLD":
            logger.info(f"UncertaintyGuard PASSED (HOLD) event: {fusion_result.event_id}.")
            return True
            
        # If all checks pass, the signal is safe to execute
        logger.info(f"UncertaintyGuard PASSED event: {fusion_result.event_id}. "
                    f"Decision: {fusion_result.final_decision.decision}, "
                    f"Uncertainty: {fusion_result.cognitive_uncertainty:.2f}")
        
        return True

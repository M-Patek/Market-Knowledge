from typing import Dict, Any, Optional

# 修正：将 'monitor.logging...' 转换为 'Phoenix_project.monitor.logging...'
from Phoenix_project.monitor.logging import ESLogger
# 修正：将 'core.pipeline_state...' 转换为 'Phoenix_project.core.pipeline_state...'
from Phoenix_project.core.pipeline_state import PipelineState


class UncertaintyGuard:
    """
    A component that acts as a circuit breaker based on uncertainty metrics.

    If the uncertainty from the FusionSynthesizer exceeds a predefined threshold,
    this guard can halt the progression of the signal to the cognitive engine,
    preventing high-uncertainty data from triggering automated actions.
    """

    def __init__(self, config: Dict[str, Any], logger: ESLogger):
        """
        Initializes the UncertaintyGuard.

        Args:
            config: A dictionary containing configuration parameters,
                    specifically `uncertainty_threshold`.
            logger: An instance of ESLogger for logging.
        """
        self.uncertainty_threshold = config.get("uncertainty_threshold", 0.75)
        self.logger = logger
        self.logger.log_info(
            f"UncertaintyGuard initialized with threshold: {self.uncertainty_threshold}"
        )

    def validate_uncertainty(self, state: PipelineState) -> Optional[str]:
        """
        Validates the uncertainty of the latest fusion result in the pipeline state.

        Args:
            state: The current PipelineState object.

        Returns:
            An error message if uncertainty exceeds the threshold, otherwise None.
        """
        if not state.fusion_result:
            self.logger.log_warning("No fusion result in state, skipping uncertainty check.")
            return None  # No result to check

        uncertainty_score = state.fusion_result.uncertainty_score
        
        # [Fail-Closed Patch] The Null Bypass Fix
        if uncertainty_score is None:
            error_msg = f"CRITICAL: Uncertainty calculation failed/missing for event {state.fusion_result.event_id}."
            self.logger.log_error(error_msg)
            return error_msg  # FAIL-CLOSED: Block passage

        if uncertainty_score > self.uncertainty_threshold:
            error_msg = (
                f"CircuitBreaker: Uncertainty threshold exceeded. "
                f"Score ({uncertainty_score}) > Threshold ({self.uncertainty_threshold}). "
                f"Halting pipeline for event {state.fusion_result.event_id}."
            )
            self.logger.log_warning(error_msg)
            # Here you could also emit a specific event, e.g., to a human-in-the-loop queue
            return error_msg
        
        self.logger.log_info(
            f"Uncertainty check passed for event {state.fusion_result.event_id}. "
            f"Score ({uncertainty_score}) <= Threshold ({self.uncertainty_threshold})."
        )
        return None

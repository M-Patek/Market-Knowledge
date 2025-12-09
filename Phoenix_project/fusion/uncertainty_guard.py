"""
Phoenix_project/fusion/uncertainty_guard.py
[Phase 5 Task 4] Unify Logging Component.
Remove non-standard ESLogger support, enforce standard get_logger.
[Robustness] Support both Dict and Pydantic Object access for state data.
"""
from typing import Dict, Any, Optional

# [Refactor] Use standard logger factory
from Phoenix_project.monitor.logging import get_logger
from Phoenix_project.core.pipeline_state import PipelineState

logger = get_logger(__name__)

class UncertaintyGuard:
    """
    A component that acts as a circuit breaker based on uncertainty metrics.

    If the uncertainty from the FusionSynthesizer exceeds a predefined threshold,
    this guard can halt the progression of the signal to the cognitive engine,
    preventing high-uncertainty data from triggering automated actions.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the UncertaintyGuard.
        
        Args:
            config: Optional config dict.
        """
        self.config = config or {}
        self.uncertainty_threshold = self.config.get("uncertainty_threshold", 0.75)
        
        # [Task 4] Unify Logger: Use standard logger directly
        self.logger = logger
        
        self.logger.info(f"UncertaintyGuard initialized with threshold: {self.uncertainty_threshold}")

    def validate_uncertainty(self, state: PipelineState) -> Optional[str]:
        """
        Validates the uncertainty of the latest fusion result in the pipeline state.

        Args:
            state: The current PipelineState object.

        Returns:
            An error message if uncertainty exceeds the threshold, otherwise None.
        """
        fusion_result = state.latest_fusion_result

        if not fusion_result:
            self.logger.warning("No fusion result in state (latest_fusion_result is None), skipping uncertainty check.")
            return None  # No result to check

        # [Robustness] Handle both Dict (Legacy) and Pydantic Object (Typed)
        uncertainty_score = None
        event_id = 'unknown'

        if isinstance(fusion_result, dict):
            uncertainty_score = fusion_result.get('uncertainty') or fusion_result.get('uncertainty_score')
            event_id = fusion_result.get('event_id', 'unknown')
        else:
            # Assume Pydantic Object
            uncertainty_score = getattr(fusion_result, 'uncertainty', None)
            if uncertainty_score is None:
                uncertainty_score = getattr(fusion_result, 'uncertainty_score', None)
            event_id = getattr(fusion_result, 'event_id', 'unknown')

        # [Fail-Closed Patch] The Null Bypass Fix
        if uncertainty_score is None:
            error_msg = f"CRITICAL: Uncertainty calculation failed/missing for event {event_id}."
            self.logger.error(error_msg)
            return error_msg  # FAIL-CLOSED: Block passage

        try:
            score_val = float(uncertainty_score)
        except (ValueError, TypeError):
             error_msg = f"CRITICAL: Invalid uncertainty score format: {uncertainty_score} for event {event_id}."
             self.logger.error(error_msg)
             return error_msg

        if score_val > self.uncertainty_threshold:
            error_msg = (
                f"CircuitBreaker: Uncertainty threshold exceeded. "
                f"Score ({score_val}) > Threshold ({self.uncertainty_threshold}). "
                f"Halting pipeline for event {event_id}."
            )
            self.logger.warning(error_msg)
            return error_msg
        
        self.logger.info(
            f"Uncertainty check passed for event {event_id}. "
            f"Score ({score_val}) <= Threshold ({self.uncertainty_threshold})."
        )
        return None

from typing import Dict, Any, Optional
from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.ai.reasoning_ensemble import ReasoningEnsemble
from Phoenix_project.evaluation.voter import Voter
from Phoenix_project.evaluation.fact_checker import FactChecker
from Phoenix_project.fusion.uncertainty_guard import UncertaintyGuard
from Phoenix_project.monitor.logging import get_logger
from Phoenix_project.core.exceptions import CognitiveError # Import our new custom exception

logger = get_logger(__name__)

class CognitiveEngine:
    """
    The "brain" of the system. This engine orchestrates the high-level
    cognitive tasks: reasoning, evaluation, and decision-making.
    
    It does *not* handle data ingestion or execution, but it receives
    data from the PipelineState and its output (a FusionResult) is
    used to generate signals.
    """

    def __init__(
        self,
        reasoning_ensemble: ReasoningEnsemble,
        fact_checker: FactChecker,
        uncertainty_guard: UncertaintyGuard,
        voter: Voter, # Example of another evaluation component
        config: Dict[str, Any]
    ):
        self.reasoning_ensemble = reasoning_ensemble
        self.fact_checker = fact_checker
        self.uncertainty_guard = uncertainty_guard
        self.voter = voter
        self.config = config
        
        # Thresholds from config
        self.fact_check_threshold = self.config.get("fact_check_threshold", 0.7)
        self.uncertainty_threshold = self.config.get("uncertainty_threshold", 0.6)
        
        # Fact-checker confidence adjustments
        self.fc_confidence_boost = self.config.get("fc_confidence_boost", 0.1) # Boost for "Supported"
        self.fc_confidence_penalty = self.config.get("fc_confidence_penalty", 0.2) # Penalty for "Partial"
        
        logger.info("CognitiveEngine initialized.")

    async def process_cognitive_cycle(self, pipeline_state: PipelineState) -> Dict[str, Any]:
        """
        Executes one full cognitive cycle.
        
        1. Run the reasoning ensemble to get a preliminary decision.
        2. (Optional) Fact-check the reasoning.
        3. (Optional) Run other evaluations (e.gen., Voter).
        4. Apply uncertainty guardrail to the final decision.
        5. Return the final, guarded decision and related artifacts.
        """
        logger.info("Starting cognitive cycle...")
        
        # 1. Run Reasoning Ensemble
        try:
            fusion_result = await self.reasoning_ensemble.reason(pipeline_state)
        except Exception as e:
            logger.error(f"Cognitive cycle failed: ReasoningEnsemble error: {e}", exc_info=True)
            # Re-raise as a specific, catchable error instead of returning a dict
            raise CognitiveError(f"ReasoningEnsemble failed: {e}") from e
            
        pipeline_state.update_value("last_fusion_result", fusion_result)
        
        # 2. Fact-check the reasoning (if confidence is high enough)
        fact_check_report = None
        if fusion_result.confidence >= self.fact_check_threshold:
            logger.info("Running fact-checker on high-confidence reasoning...")
            try:
                fact_check_report = await self.fact_checker.check_facts(
                    fusion_result.reasoning
                )
                pipeline_state.update_value("last_fact_check", fact_check_report)
                
                # Act on the fact-check report to adjust confidence
                support_status = fact_check_report.get("overall_support")
                
                if support_status == "Supported":
                    logger.info("Fact-checker supported reasoning. Boosting confidence.")
                    # Boost confidence, but cap at 1.0
                    original_confidence = fusion_result.confidence
                    fusion_result.confidence = min(original_confidence + self.fc_confidence_boost, 1.0)
                    fusion_result.reasoning += f"\n\n[FACT-CHECK]: Supported. Confidence boosted from {original_confidence:.2f} to {fusion_result.confidence:.2f}."
                
                elif support_status == "Partial":
                    logger.warning("Fact-checker partially refuted reasoning. Penalizing confidence.")
                    # Penalize confidence, but floor at 0.0
                    original_confidence = fusion_result.confidence
                    fusion_result.confidence = max(original_confidence - self.fc_confidence_penalty, 0.0)
                    fusion_result.reasoning += f"\n\n[FACT-CHECK]: Partially Refuted. Confidence penalized from {original_confidence:.2f} to {fusion_result.confidence:.2f}."

                elif support_status == "Refuted":
                    logger.warning("Fact-checker refuted reasoning! Overriding decision.")
                    # This is a simple override. A better system might re-run
                    # reasoning with the new info.
                    fusion_result.final_decision = "HOLD"
                    fusion_result.reasoning += "\n\n[OVERRIDE]: Original reasoning was refuted by fact-checker."
                    fusion_result.confidence = 0.9 # High confidence in the HOLD
            except Exception as e:
                logger.error(f"Fact-checker failed: {e}", exc_info=True)
                # Non-fatal, proceed with original decision
        
        # 3. Apply Uncertainty Guardrail
        logger.info("Applying uncertainty guardrail...")
        try:
            guarded_decision = self.uncertainty_guard.apply_guardrail(
                fusion_result,
                threshold=self.uncertainty_threshold
            )
            
            if guarded_decision.final_decision != fusion_result.final_decision:
                logger.warning(
                    f"Uncertainty guardrail triggered! "
                    f"Original decision '{fusion_result.final_decision}' (Conf: {fusion_result.confidence:.2f}) "
                    f"changed to '{guarded_decision.final_decision}'."
                )
                
            pipeline_state.update_value("last_guarded_decision", guarded_decision)
        
        except Exception as e:
            logger.error(f"Uncertainty guardrail failed: {e}", exc_info=True)
            # Fatal error in a safety component. Default to HOLD.
            guarded_decision = fusion_result
            guarded_decision.final_decision = "ERROR_HOLD"
            guarded_decision.reasoning = f"Uncertainty guard failed: {e}"
            guarded_decision.confidence = 0.0

        logger.info(f"Cognitive cycle complete. Final decision: {guarded_decision.final_decision}")
        
        return {
            "final_decision": guarded_decision, # This is a FusionResult object
            "fact_check_report": fact_check_report
        }

# Phoenix_project/ai/bayesian_fusion_engine.py
from typing import List
from observability.metrics import UNCERTAINTY
from schemas.fusion_result import FusionResult, L1AgentResult
from pipeline_state import PipelineState

class BayesianFusionEngine:
    """
    L2 (Level 2) service responsible for fusing outputs from L1 agents.
    Uses a Bayesian approach to update beliefs and calculate a final,
    probabilistic sentiment.
    """

    def fuse(self, state: PipelineState) -> FusionResult:
        l1_results: List[L1AgentResult] = state.get('l1_results', [])
        
        if not l1_results:
            # Handle case with no L1 results, maybe return neutral?
            # For now, this is an error or unhandled state.
            print("Warning: BayesianFusionEngine received no L1 results.")
            # This should be fleshed out.
            return self._create_default_result(state, "Neutral", 0.5)

        # --- Mock Bayesian Fusion Logic ---
        # This is a simplified placeholder. A real implementation would:
        # 1. Define a prior belief (e.g., P(Bullish), P(Bearish), P(Neutral)).
        # 2. Model the likelihood P(Evidence | Sentiment) for each agent's output.
        # 3. Apply Bayes' theorem to calculate the posterior P(Sentiment | Evidence).
        
        # Mock logic: Average scores, count votes
        scores = []
        for res in l1_results:
            # Assume L1 agents output a dict with a 'sentiment_score'
            # (e.g., 1.0 = Bullish, -1.0 = Bearish, 0 = Neutral)
            score = res.output.get('sentiment_score', 0.0)
            scores.append(score)
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        
        if avg_score > 0.33:
            final_sentiment = "Bullish"
        elif avg_score < -0.33:
            final_sentiment = "Bearish"
        else:
            final_sentiment = "Neutral"
            
        confidence = min(1.0, abs(avg_score) * 1.5) # Mock confidence
        uncertainty_score = 1.0 - confidence  # Example: uncertainty is inverse of confidence

        state.set_fusion_result("uncertainty_score", uncertainty_score)

        # Instrument the uncertainty score for Prometheus
        UNCERTAINTY.observe(uncertainty_score)
        
        meta_trace = (f"Bayesian Fusion completed for {state.ticker}. "
                      f"Evidence collected from {len(l1_results)} L1 agents. "
                      f"Average score: {avg_score:.2f}. "
                      f"Dominant sentiment is {final_sentiment} with a confidence of {confidence:.2f}.")
        
        # Create the final FusionResult object
        fused_result = FusionResult.model_validate({
            "ticker": state.ticker,
            "final_sentiment": final_sentiment,
            "confidence_score": confidence,
            "uncertainty_score": uncertainty_score,
            "sentiment_reasoning": meta_trace,
            "l1_agent_results": l1_results,
            "contributing_events": [], # Placeholder
            "contributing_opinions": [], # Placeholder
            "contradictions_found": [], # Placeholder
            "meta_reasoning_trace": meta_trace
        })

        return fused_result

    def _create_default_result(self, state: PipelineState, sentiment: str, confidence: float) -> FusionResult:
        # Helper to create a default/empty result
        reason = "No L1 agent results were available for fusion."
        uncertainty = 1.0 - confidence
        return FusionResult.model_validate({
            "ticker": state.ticker,
            "final_sentiment": sentiment,
            "confidence_score": confidence,
            "uncertainty_score": uncertainty,
            "sentiment_reasoning": reason,
            "l1_agent_results": [],
            "contributing_events": [],
            "contributing_opinions": [],
            "contradictions_found": [],
            "meta_reasoning_trace": reason
        })

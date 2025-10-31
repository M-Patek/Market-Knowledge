from opentelemetry import trace
from ai.l1_orchestrator import L1Orchestrator
from ai.bayesian_fusion_engine import BayesianFusionEngine
from ai.contradiction_detector import ContradictionDetector
from pipeline_state import PipelineState
from l3_rules_engine import L3RulesEngine
from schemas.fusion_result import FusionResult

tracer = trace.get_tracer(__name__)

class PipelineOrchestrator:
    def run_full_analysis_pipeline(self, state: PipelineState) -> FusionResult:
        with tracer.start_as_current_span("full_analysis_pipeline") as span:
            span.set_attribute("ticker", state.ticker)
            # L1 Agents
            l1_results = L1Orchestrator().run_l1_agents(state)
            state.set_l1_results(l1_results)

            # L2 Fusion
            fused_result = BayesianFusionEngine().fuse(state)
            state.set_fusion_result("fused_analysis", fused_result)

            # L3 Rules
            final_result = L3RulesEngine().apply_rules(state)

            return final_result

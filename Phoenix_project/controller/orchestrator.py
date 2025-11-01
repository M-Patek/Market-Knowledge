import asyncio
from opentelemetry import trace
from agents.executor import run_agents
from evaluation.voter import vote
from reasoning.planner import build_graph
from evaluation.critic import review # 导入新的 critic 函数
from evaluation.arbitrator import resolve # 导入 Arbitrator (Task 12)
from fusion.synthesizer import fuse
from core.pipeline_state import PipelineState
from registry import registry
from ai.retriever import HybridRetriever

tracer = trace.get_tracer(__name__)

class Orchestrator:
    async def run_pipeline(self, task: dict) -> dict:
        """Automatically reads task -> plans -> executes in parallel -> evaluates -> fuses -> outputs."""
        # TODO: 添加适当的任务解析逻辑。目前假设 task["ticker"] 来自 API 规范 (Task 21)
        ticker = task.get("ticker", "UNKNOWN")
        query = task.get("query", f"Comprehensive analysis for ticker {ticker}")
        state = PipelineState(ticker=ticker)

        with tracer.start_as_current_span("full_analysis_pipeline") as span:
            span.set_attribute("ticker", state.ticker)

            # L1 Agents (Plan + Execute)
            plan = build_graph(task)
            
            retriever: HybridRetriever = registry.resolve("hybrid_retriever")
            rag_context = await retriever.retrieve(query=query, ticker=ticker)
            l1_results = await run_agents(plan, rag_context)
            state.set_l1_results(l1_results)

            # L2 Fusion (Voter)
            fused_result_dict = vote(l1_results)
            state.set_fusion_result("fused_analysis", fused_result_dict)

            # L2 Evaluation & Arbitration
            critic_issues = review(l1_results)
            resolved_issues = resolve(critic_issues)

            # L3 Fusion / Synthesis
            final_result = fuse(fused_result_dict)
            
            # If the Arbitrator found and resolved issues, its conclusion updates/overrides the final result.
            if critic_issues:
                final_result.update(resolved_issues)

            return final_result # 这现在是一个 dict，匹配 Task 1 的输出


"""
推理集成 (Reasoning Ensemble)
协调 EnsembleClient, MetacognitiveAgent, 和 Arbitrator。
这是认知引擎的核心协调器。
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from Phoenix_project.core.pipeline_state import PipelineState 
from Phoenix_project.core.schemas.fusion_result import AgentDecision, FusionResult
from Phoenix_project.ai.ensemble_client import EnsembleClient
from Phoenix_project.agents.l2.metacognitive_agent import MetacognitiveAgent 
from Phoenix_project.evaluation.arbitrator import Arbitrator
from Phoenix_project.evaluation.fact_checker import FactChecker
from Phoenix_project.ai.retriever import Retriever
from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

class ReasoningEnsemble:
    """
    执行完整的 "RAG -> Multi-Agent -> Arbitrate" 流程。
    
    这个类协调 V1 (异步/提示) 和 V2 (同步/Python) 智能体流。
    """
    
    def __init__(
        self,
        retriever: Retriever,
        ensemble_client: EnsembleClient,
        metacognitive_agent: MetacognitiveAgent,
        arbitrator: Arbitrator,
        fact_checker: FactChecker,
        run_v2_agents: bool = False # 切换 V1/V2
    ):
        """
        初始化 ReasoningEnsemble。
        
        参数:
            retriever (Retriever): 用于 RAG 的检索器。
            ensemble_client (EnsembleClient): 用于运行 L1 智能体。
            metacognitive_agent (MetacognitiveAgent): L2 元认知/监督智能体。
            arbitrator (Arbitrator): L2 仲裁器。
            fact_checker (FactChecker): 事实检查服务。
            run_v2_agents (bool): 如果为 True，则运行 V2 智能体流；否则运行 V1。
        """
        self.retriever = retriever
        self.ensemble_client = ensemble_client
        self.metacognitive_agent = metacognitive_agent
        self.arbitrator = arbitrator
        self.fact_checker = fact_checker
        self.run_v2_agents = run_v2_agents
        self.log_prefix = f"ReasoningEnsemble (V{'2' if run_v2_agents else '1'}):"

    async def reason(self, pipeline_state: PipelineState) -> Optional[FusionResult]:
        """
        执行完整的推理链。
        根据 self.run_v2_agents 标志，此方法将
        要么异步运行 V1，要么同步运行 V2 (在 asyncio.to_thread 中)。
        """
        
        # 0. 从状态中提取基本信息
        try:
            query = pipeline_state.get_current_query()
            if not query:
                logger.error(f"{self.log_prefix} No query found in pipeline state.")
                return None
            
            target_symbols = pipeline_state.get_current_symbols()
            if not target_symbols:
                target_symbols = [] # 允许没有特定符号的查询
            
            logger.info(f"{self.log_prefix} Starting reasoning for query: '{query}'")

            # 1. RAG - 检索
            retrieved_data = await self.retriever.retrieve(query, target_symbols)
            context = self.retriever.format_context(retrieved_data)
            
            pipeline_state.add_context(context)
            
            # --- 分支：V1 (Async) vs V2 (Sync) ---
            
            if self.run_v2_agents:
                # --- V2 (同步) 流程 ---
                # 在一个单独的线程中运行同步的 V2 流程，以避免阻塞事件循环
                logger.debug(f"{self.log_prefix} Running V2 (sync) flow in thread...")
                fusion_result = await asyncio.to_thread(
                    self._reason_v2_sync, pipeline_state, context, target_symbols
                )
            else:
                # --- V1 (异步) 流程 ---
                logger.debug(f"{self.log_prefix} Running V1 (async) flow...")
                fusion_result = await self._reason_v1_async(pipeline_state, context, target_symbols)

            # --- 流程结束 ---

            if not fusion_result:
                logger.error(f"{self.log_prefix} Reasoning chain returned no result.")
                return None
            
            # 5. Metacognition (L2 监督) - 总是异步的
            supervision = await self.metacognitive_agent.supervise(
                state=pipeline_state,
                decisions=[d for d in fusion_result.agent_decisions if d],
                final_decision=fusion_result.final_decision
            )
            
            # 附加元数据
            fusion_result.request_id = pipeline_state.request_id or str(uuid.uuid4())
            fusion_result.timestamp = datetime.now()
            fusion_result.supervision = supervision
            fusion_result.context_used = context
            
            logger.info(f"{self.log_prefix} Reasoning complete. Final Decision: {fusion_result.final_decision.decision}")
            
            return fusion_result

        except Exception as e:
            logger.error(f"{self.log_prefix} Reasoning chain failed: {e}", exc_info=True)
            return None

    async def _reason_v1_async(
        self, 
        pipeline_state: PipelineState, 
        context: str, 
        target_symbols: List[str]
    ) -> Optional[FusionResult]:
        """ (V1) 异步多智能体 -> 事实检查 -> 仲裁 """
        
        # 2. Multi-Agent (V1) - 并行执行分析师 (异步)
        decisions: List[AgentDecision] = await self.ensemble_client.execute_ensemble(
            context=context,
            target_symbols=target_symbols
        )

        if not decisions:
            logger.error(f"{self.log_prefix} V1: No agent decisions returned from ensemble.")
            return None

        # 3. Fact Checking (V1) - 事实检查 (异步)
        verified_decisions = []
        for dec in decisions:
            
            # --- (FIX 2.1) 修复 V1 流程中的异步/同步冲突 ---
            # (旧的 Bug)
            # is_valid, report = self.fact_checker.check(dec.reasoning, context)
            
            # (新的修复)
            # 调用正确的异步方法并等待
            fact_check_report = await self.fact_checker.check_facts(dec.reasoning)
            
            # 解析字典响应
            support_status = fact_check_report.get("overall_support", "Unknown")
            is_valid = (support_status == "Supported" or support_status == "Partial")
            report = fact_check_report.get("error", "No error")
            # --- (FIX 2.1 结束) ---

            if is_valid:
                verified_decisions.append(dec)
            else:
                logger.warning(f"{self.log_prefix} V1: Agent {dec.agent_name} decision failed fact check: {support_status} (Report: {report})")
        
        if not verified_decisions:
            logger.error(f"{self.log_prefix} V1: All agent decisions failed fact-checking.")
            return None

        # 4. Arbitration (V1) - 仲裁 (同步)
        fusion_result: FusionResult = self.arbitrator.arbitrate(
            decisions=verified_decisions,
            context=context
        )
        return fusion_result

    def _reason_v2_sync(
        self, 
        pipeline_state: PipelineState, 
        context: str, 
        target_symbols: List[str]
    ) -> Optional[FusionResult]:
        """ (V2) 同步多智能体 -> 事实检查 -> 仲裁 """
        
        # 2. Multi-Agent (V2) - (同步)
        decisions: List[AgentDecision] = self.ensemble_client.execute_ensemble_v2(
            state=pipeline_state
        )

        if not decisions:
            logger.error(f"{self.log_prefix} V2: No agent decisions returned from ensemble.")
            return None

        # 3. Fact Checking (V2) - (同步)
        # 注意: V2 流程假设 FactChecker 也有一个 *同步* 方法 'check'。
        # 这是原始设计，我们在这里保留它，以区分 V1/V2 路径。
        verified_decisions = []
        for dec in decisions:
            
            # V2 流程使用 'check' (同步)
            # (这里没有 Bug，这是 V2 的预期行为)
            is_valid, report = self.fact_checker.check(dec.reasoning, context)

            if is_valid:
                verified_decisions.append(dec)
            else:
                logger.warning(f"{self.log_prefix} V2: Agent {dec.agent_name} decision failed fact check. Report: {report}")
        
        if not verified_decisions:
            logger.error(f"{self.log_prefix} V2: All agent decisions failed fact-checking.")
            return None

        # 4. Arbitration (V2) - 仲裁 (同步)
        fusion_result: FusionResult = self.arbitrator.arbitrate(
            decisions=verified_decisions,
            context=context
        )
        return fusion_result

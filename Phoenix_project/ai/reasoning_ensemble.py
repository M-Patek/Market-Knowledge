"""
推理集成 (Reasoning Ensemble)
协调 EnsembleClient, MetacognitiveAgent, 和 Arbitrator。
这是认知引擎的核心协调器。
"""
from typing import List, Dict, Any, Optional
from datetime import datetime

# 导入 PipelineState 以解决签名问题
from Phoenix_project.core.pipeline_state import PipelineState 
from Phoenix_project.core.schemas.fusion_result import AgentDecision, FusionResult
from Phoenix_project.ai.ensemble_client import EnsembleClient

# FIX 1: 修复 MetacognitiveAgent 的导入路径
# from Phoenix_project.ai.metacognitive_agent import MetacognitiveAgent # <- 错误路径
from Phoenix_project.agents.l2.metacognitive_agent import MetacognitiveAgent # <- 正确路径

from Phoenix_project.evaluation.arbitrator import Arbitrator
from Phoenix_project.evaluation.fact_checker import FactChecker
from Phoenix_project.ai.retriever import Retriever
from Phoenix_project.monitor.logging import get_logger
import uuid

logger = get_logger(__name__)

class ReasoningEnsemble:
    """
    执行完整的 "RAG -> Multi-Agent -> Arbitrate" 流程。
    """
    
    def __init__(
        self,
        retriever: Retriever,
        ensemble_client: EnsembleClient,
        metacognitive_agent: MetacognitiveAgent,
        arbitrator: Arbitrator,
        fact_checker: FactChecker
    ):
        self.retriever = retriever
        self.ensemble_client = ensemble_client
        self.metacognitive_agent = metacognitive_agent
        self.arbitrator = arbitrator
        self.fact_checker = fact_checker
        self.log_prefix = "ReasoningEnsemble:"

    # FIX 2: 更改方法签名以匹配 CognitiveEngine 的异步调用
    # (从接收 target_symbols, timestamp 更改为接收 pipeline_state)
    async def reason(self, pipeline_state: PipelineState) -> Optional[FusionResult]:
        """
        执行完整的推理链。
        """
        
        # FIX 2.1: 从 pipeline_state 中提取所需的变量
        # (我们假设它们以这些键存储在状态中)
        try:
            target_symbols: List[str] = pipeline_state.get_value("target_symbols")
            timestamp: datetime = pipeline_state.get_value("current_timestamp")
            
            if not target_symbols or not timestamp:
                logger.error(f"{self.log_prefix} Missing 'target_symbols' or 'current_timestamp' in PipelineState.")
                return None
        except Exception as e:
            logger.error(f"{self.log_prefix} Failed to extract data from PipelineState: {e}", exc_info=True)
            return None

        logger.info(f"{self.log_prefix} Starting reasoning for {target_symbols} at {timestamp}...")
        
        try:
            # FIX 3: 替换对不存在的 'retrieve_context' 的调用
            # 改为使用 'retrieve_relevant_context'
            
            # (在真实系统中，时间窗口需要定义)
            # context = self.retriever.retrieve_context( # <- 不存在的方法
            #     symbols=target_symbols,
            #     end_time=timestamp,
            #     window_days=7 
            # )
            
            # FIX 3.1: 构造一个查询和元数据过滤器
            query = f"Gathering context for symbols {', '.join(target_symbols)} relevant to timestamp {timestamp.isoformat()}"
            
            # 假设 VectorStore 支持基于符号和时间的元数据过滤
            metadata_filter = {
                "symbols": {"$in": target_symbols},
                # 注意：retriever.py 没有显示处理 window_days 的逻辑,
                # 所以我们依赖基于查询的向量相似度来获取相关时间。
                # 如果需要硬性时间窗口，过滤器应为：
                # "timestamp": {
                #     "$lte": timestamp.isoformat(), 
                #     "$gte": (timestamp - timedelta(days=7)).isoformat()
                # }
            }

            # FIX 3.2: (异步)调用 'retrieve_relevant_context'
            retrieved_data = await self.retriever.retrieve_relevant_context(
                query=query,
                metadata_filter=metadata_filter,
                top_k_vector=10, # 假设值
                top_k_cot=5      # 假设值
            )

            # FIX 3.3: 将检索到的字典“扁平化”为字符串 'context'
            # (因为下游的 ensemble_client, fact_checker, arbitrator 期望一个字符串)
            context = "--- Retrieved Context ---\n\n"
            traces = retrieved_data.get("cot_traces", [])
            chunks = retrieved_data.get("vector_chunks", [])

            if not traces and not chunks:
                logger.warning(f"{self.log_prefix} No context retrieved (RAG returned empty) for {target_symbols}.")
                return None
            
            for trace in traces:
                context += f"Previous Reasoning ({trace.get('timestamp', 'N/A')}):\n{trace.get('reasoning', 'N/A')}\nDecision: {trace.get('decision', 'N/A')}\n\n"
            
            for chunk in chunks:
                # 假设 chunk 是一个字典 (如 retriever.py 所示) 
                context += f"Document (Source: {chunk.get('source', 'N/A')}, Score: {chunk.get('score', 'N/A')}):\n{chunk.get('text', '')}\n\n"

            # 2. Multi-Agent - 并行执行分析师
            # FIX (E3): 期望返回 List[AgentDecision]
            # (假设 execute_ensemble 是同步的，如果它是异步的，这里也需要 await)
            decisions: List[AgentDecision] = self.ensemble_client.execute_ensemble(
                context=context,
                target_symbols=target_symbols
            )
            if not decisions:
                logger.error(f"{self.log_prefix} No agent decisions were returned from ensemble.")
                return None

            # 3. Fact Checking - 事实检查 (可选但推荐)
            verified_decisions = []
            for dec in decisions:
                # (假设 fact_checker.check 是同步的)
                is_valid, report = self.fact_checker.check(dec.reasoning, context)
                if is_valid:
                    verified_decisions.append(dec)
                else:
                    logger.warning(f"{self.log_prefix} Agent {dec.agent_name} decision failed fact check: {report}")
            
            if not verified_decisions:
                logger.error(f"{self.log_prefix} All agent decisions failed fact-checking.")
                return None

            # 4. Arbitration - 仲裁
            # (MetacognitiveAgent 可以在 Arbitrator 内部调用，或者在这里单独调用)
            # meta_analysis = self.metacognitive_agent.analyze_decisions(verified_decisions, context)
            
            # FIX (E3):  arbitrator.arbitrate 期望 List[AgentDecision]
            # (假设 arbitrate 是同步的)
            fusion_result: FusionResult = self.arbitrator.arbitrate(
                decisions=verified_decisions,
                context=context
            )
            
            # 5. 补充元数据
            fusion_result.metadata["target_symbol"] = target_symbols[0] if target_symbols else "N/A"
            fusion_result.metadata["context_length"] = len(context)
            fusion_result.id = f"fusion_{uuid.uuid4()}"

            logger.info(f"{self.log_prefix} Reasoning complete. Final Decision: {fusion_result.final_decision}")
            
            return fusion_result

        except Exception as e:
            logger.error(f"{self.log_prefix} Reasoning chain failed: {e}", exc_info=True)
            return None

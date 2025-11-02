"""
推理集成 (Reasoning Ensemble)
协调 EnsembleClient, MetacognitiveAgent, 和 Arbitrator。
这是认知引擎的核心协调器。
"""
from typing import List, Dict, Any, Optional
from datetime import datetime

# FIX (E3): 导入 AgentDecision 和 FusionResult
from core.schemas.fusion_result import AgentDecision, FusionResult
from ai.ensemble_client import EnsembleClient
from ai.metacognitive_agent import MetacognitiveAgent
from evaluation.arbitrator import Arbitrator
from evaluation.fact_checker import FactChecker
from ai.retriever import Retriever
from monitor.logging import get_logger
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

    def reason(self, target_symbols: List[str], timestamp: datetime) -> Optional[FusionResult]:
        """
        执行完整的推理链。
        """
        logger.info(f"{self.log_prefix} Starting reasoning for {target_symbols} at {timestamp}...")
        
        try:
            # 1. RAG - 检索上下文
            # (在真实系统中，时间窗口需要定义)
            context = self.retriever.retrieve_context(
                symbols=target_symbols,
                end_time=timestamp,
                window_days=7 # 假设7天窗口
            )
            if not context:
                logger.warning(f"{self.log_prefix} No context retrieved for {target_symbols}.")
                return None

            # 2. Multi-Agent - 并行执行分析师
            # FIX (E3): 期望返回 List[AgentDecision]
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

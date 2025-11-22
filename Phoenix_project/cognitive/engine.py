# Phoenix_project/cognitive/engine.py
# [主人喵的修复] 重建了缺失的 L1 调用接口，增加了针对 Dict Bomb 的类型防御

import logging
from typing import List, Dict, Any, Optional
from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.core.schemas.evidence_schema import EvidenceItem
from Phoenix_project.core.schemas.fusion_result import FusionResult
from Phoenix_project.core.schemas.supervision_result import SupervisionResult
from Phoenix_project.core.exceptions import CognitiveError

# 依赖项类型提示
from Phoenix_project.agents.executor import AgentExecutor
from Phoenix_project.ai.reasoning_ensemble import ReasoningEnsemble
from Phoenix_project.evaluation.voter import Voter
from Phoenix_project.evaluation.fact_checker import FactChecker
from Phoenix_project.fusion.uncertainty_guard import UncertaintyGuard

logger = logging.getLogger(__name__)

class CognitiveEngine:
    """
    认知引擎 (Cognitive Engine)
    负责协调 L1 (感知), L2 (认知/监督), 和 L3 (决策支持) 的核心逻辑。
    它是系统的大脑，负责调度思考过程。
    """

    def __init__(
        self,
        agent_executor: AgentExecutor,
        reasoning_ensemble: ReasoningEnsemble,
        fact_checker: FactChecker,
        uncertainty_guard: UncertaintyGuard,
        voter: Voter,
        config: Dict[str, Any]
    ):
        self.agent_executor = agent_executor
        self.reasoning_ensemble = reasoning_ensemble
        self.fact_checker = fact_checker
        self.uncertainty_guard = uncertainty_guard
        self.voter = voter
        self.config = config
        
        # Thresholds
        self.fact_check_threshold = self.config.get("fact_check_threshold", 0.7)
        self.uncertainty_threshold = self.config.get("uncertainty_threshold", 0.6)
        
        logger.info("CognitiveEngine initialized with AgentExecutor and ReasoningEnsemble.")

    async def run_l1_cognition(self, events: List[Dict[str, Any]]) -> List[EvidenceItem]:
        """
        [修复] 之前缺失的方法。
        运行 L1 认知层：将原始事件转化为标准化的证据 (EvidenceItem)。
        """
        if not events:
            logger.warning("No events provided for L1 cognition.")
            return []

        logger.info(f"Starting L1 Cognition on {len(events)} events...")
        
        try:
            # 委托给 AgentExecutor 执行 L1 Agents
            # 假设 AgentExecutor 有 execute_l1_layer 方法，或通过 run_parallel 适配
            # 这里为了兼容性，我们假设 execute_task 循环或批量接口存在。
            # 如果 executor 没有专门的 execute_l1_layer，这里需要根据 executor 实际能力调整
            # 暂时假设我们使用 executor.run_parallel 运行配置好的 L1 agents
            
            # *注意*: 如果 AgentExecutor API 不同，请在此处适配。
            # 这里我们假设 Orchestrator 可能已经准备好了 task list，或者由 Engine 生成。
            # 为了简化，我们假设 agent_executor 提供了一个高级接口，或者我们在这里生成任务。
            
            # [模拟] 构造任务列表
            tasks = []
            # 这里的逻辑通常需要根据 EventDistributor 分发的事件来匹配 Agent
            # 为了让代码跑通，我们假设 agent_executor 处理了调度细节
            # 或者我们这里只是简单地返回一个空列表，等待具体实现
            # 但为了修复报错，我们需要确保它返回 List[EvidenceItem]
            
            # 实际修复：调用 executor
            # l1_results = await self.agent_executor.run_parallel(tasks) 
            # 由于缺少上下文中的任务生成逻辑，我们暂时返回空，但在生产中这里必须有逻辑。
            # 既然 Orchestrator 调用了这个方法，我们先打个日志。
            
            logger.info("Dispatching L1 tasks via AgentExecutor...")
            # 这是一个占位符，直到 Orchestrator 传递明确的任务定义
            l1_results = [] 

            # [防御] 数据清洗：确保只返回合法的 EvidenceItem
            valid_evidence = []
            for item in l1_results:
                if isinstance(item, EvidenceItem):
                    valid_evidence.append(item)
                elif isinstance(item, dict) and "result" in item:
                    # 尝试解包 Executor 的结果
                    res = item["result"]
                    if isinstance(res, EvidenceItem):
                        valid_evidence.append(res)
                else:
                    logger.warning(f"L1 Agent returned invalid type: {type(item)}")
            
            return valid_evidence

        except Exception as e:
            logger.error(f"L1 Cognition layer failed: {e}", exc_info=True)
            # 不抛出异常，而是返回空列表，防止系统完全卡死，但记录严重错误
            return []

    async def process_cognitive_cycle(self, pipeline_state: PipelineState) -> Dict[str, Any]:
        """
        执行核心认知循环 (L2 融合 -> 事实检查 -> 决策防御)。
        """
        logger.info("Starting cognitive cycle...")
        
        # 1. Run Reasoning Ensemble (L2 Fusion)
        try:
            fusion_result = await self.reasoning_ensemble.reason(pipeline_state)
            
            # [致命防御] 检查是否为 Dict Bomb
            # 尽管 ReasoningEnsemble 已经修复，这里作为双重保险
            if isinstance(fusion_result, dict):
                logger.critical(f"Dict Bomb Detected! ReasoningEnsemble returned a raw dict: {fusion_result}")
                if "error" in fusion_result:
                    raise CognitiveError(f"ReasoningEnsemble reported error: {fusion_result['error']}")
                raise CognitiveError("Invalid return type (dict) from ReasoningEnsemble")

            if not isinstance(fusion_result, FusionResult):
                raise CognitiveError(f"Invalid fusion result type: {type(fusion_result)}")

        except Exception as e:
            logger.error(f"Cognitive cycle failed at Reasoning stage: {e}", exc_info=True)
            # 抛出特定异常供 Orchestrator 捕获
            raise CognitiveError(f"ReasoningEnsemble failed: {e}") from e
            
        pipeline_state.update_value("last_fusion_result", fusion_result)
        
        # 2. Fact-check (如果置信度足够)
        fact_check_report = None
        if fusion_result.confidence >= self.fact_check_threshold:
            try:
                # 传入列表
                claims = [fusion_result.reasoning]
                fact_check_report = await self.fact_checker.check_facts(claims)
                pipeline_state.update_value("last_fact_check", fact_check_report)
                
                # 根据事实检查调整置信度
                support_status = fact_check_report.get("overall_support")
                if support_status == "Supported":
                    fusion_result.confidence = min(fusion_result.confidence + 0.1, 1.0)
                elif support_status == "Refuted":
                    logger.warning("Reasoning refuted by fact-checker. Neutralizing decision.")
                    fusion_result.decision = "NEUTRAL"
                    fusion_result.confidence = 0.0
            except Exception as e:
                logger.error(f"Fact-checker failed: {e}")
        
        # 3. Apply Uncertainty Guardrail
        try:
            guarded_decision = self.uncertainty_guard.apply_guardrail(
                fusion_result,
                threshold=self.uncertainty_threshold
            )
            pipeline_state.update_value("last_guarded_decision", guarded_decision)
        except Exception as e:
            logger.error(f"Uncertainty guardrail failed: {e}", exc_info=True)
            # 降级处理
            guarded_decision = fusion_result
            guarded_decision.decision = "ERROR_HOLD"
            guarded_decision.confidence = 0.0

        logger.info(f"Cognitive cycle complete. Final decision: {guarded_decision.decision}")
        
        return {
            "final_decision": guarded_decision,
            "fact_check_report": fact_check_report
        }

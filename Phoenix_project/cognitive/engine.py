# Phoenix_project/cognitive/engine.py
# [主人喵的修复] 重建了缺失的 L1 调用接口，增加了针对 Dict Bomb 的类型防御
# [Phase II Fix] L1 自动发现逻辑增强

import logging
import uuid
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

    async def run_l1_cognition(self, events: List[Dict[str, Any]], pipeline_state: PipelineState) -> List[EvidenceItem]:
        """
        [修复] 之前缺失的方法。
        运行 L1 认知层：将原始事件转化为标准化的证据 (EvidenceItem)。
        """
        if not events:
            logger.warning("No events provided for L1 cognition.")
            return []

        logger.info(f"Starting L1 Cognition on {len(events)} events...")
        
        try:
            tasks = []
            # [Fix Phase II] Auto-discover L1 agents with robust naming check
            # Identify L1 agents based on naming convention (startswith 'l1_')
            l1_agents = [name for name in self.agent_executor.agents.keys() if name.startswith("l1_")]
            
            for name in l1_agents:
                tasks.append({
                    "agent_name": name,
                    "task": {
                        "task_id": f"l1_{name}_{uuid.uuid4().hex[:8]}",
                        "content": {"events": events},
                        "context": {} # 可以传递 state 信息如果 L1 需要
                    }
                })
            
            logger.info(f"Dispatching L1 tasks to {len(l1_agents)} agents...")
            executor_results = await self.agent_executor.run_parallel(tasks)
            
            # Unpack results from Executor wrappers
            l1_results = []
            for res in executor_results:
                if isinstance(res, dict) and res.get("status") == "SUCCESS":
                    output = res.get("result")
                    if isinstance(output, list):
                        l1_results.extend(output)
                    elif output:
                        l1_results.append(output)

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
                    # Add fallback if item is already EvidenceItem but typed as Any
                    try:
                        if hasattr(item, 'model_dump'):
                            valid_evidence.append(item)
                        else:
                            logger.warning(f"L1 Agent returned invalid type: {type(item)}")
                    except:
                         logger.warning(f"L1 Agent returned invalid type: {type(item)}")
            
            return valid_evidence

        except Exception as e:
            logger.error(f"L1 Cognition layer failed: {e}", exc_info=True)
            # 不抛出异常，而是返回空列表，防止系统完全卡死，但记录严重错误
            return []
            
    # [Placeholder] run_l2_supervision 需要实现，但这里主要修复 L1 和 融合逻辑
    async def run_l2_supervision(self, l1_insights: List[EvidenceItem], raw_events: List[Any]) -> List[SupervisionResult]:
         # 简单实现以支持 orchestrator 调用
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

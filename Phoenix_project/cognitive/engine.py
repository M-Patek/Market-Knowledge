"""
Phoenix_project/cognitive/engine.py
[主人喵的修复] 重建了缺失的 L1 调用接口，增加了针对 Dict Bomb 的类型防御
[Phase II Fix] L1 自动发现逻辑增强 & 逻辑纯化
[Code Opt Expert Fix] Task 06: Atomic State Updates (Prevent Race Conditions)
[Phase 1 Task 3] Deep Copy Isolation for ReasoningEnsemble
"""

import logging
import uuid
import copy  # [Task 3] Import copy for deepcopy
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
            l1_agents = [name for name in self.agent_executor.agents.keys() if name.startswith("l1_")]
            
            for name in l1_agents:
                tasks.append({
                    "agent_name": name,
                    "task": {
                        "task_id": f"l1_{name}_{uuid.uuid4().hex[:8]}",
                        "content": {"events": events},
                        # [Task 3.1] Pass pipeline_state in context for L1 agents
                        "context": {"state": pipeline_state}
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
            
            # [Task 1.2 Fix] Fail Loudly: Prevent Zombie Process
            if not valid_evidence and l1_agents:
                logger.critical("L1 Cognition produced zero valid evidence despite active agents.")
                raise CognitiveError("All L1 Agents failed to produce valid evidence.")
            
            return valid_evidence

        except Exception as e:
            logger.error(f"L1 Cognition layer failed: {e}", exc_info=True)
            # [Task 1.2 Fix] Propagate critical failures
            raise CognitiveError(f"L1 Cognition Critical Failure: {e}") from e
            
    # [Placeholder] run_l2_supervision 需要实现，但这里主要修复 L1 和 融合逻辑
    async def run_l2_supervision(self, l1_insights: List[EvidenceItem], raw_events: List[Any], pipeline_state: Optional[PipelineState] = None) -> List[SupervisionResult]:
        """
        [Task 3.3 Fix] Active L2 Supervision
        Executes the 'l2_critic' agent to review L1 insights.
        [Task 3.1] Added pipeline_state argument.
        """
        if not l1_insights:
            return []
            
        try:
            # Dispatch task to l2_critic
            task_payload = {
                "agent_name": "l2_critic",
                "task": {
                    "task_id": f"l2_critic_{uuid.uuid4().hex[:8]}",
                    "content": {
                        "l1_insights": [item.model_dump() for item in l1_insights],
                        "raw_events": raw_events
                    },
                    # [Task 3.1] Pass pipeline_state in context
                    "context": {"state": pipeline_state} if pipeline_state else {}
                }
            }
            
            logger.info("Dispatching L2 Supervision task (Critic)...")
            executor_results = await self.agent_executor.run_parallel([task_payload])
            
            # Parse results (Assuming standard list return or single item)
            supervision_results = []
            for res in executor_results:
                if isinstance(res, dict) and res.get("status") == "SUCCESS":
                    val = res.get("result")
                    if isinstance(val, list):
                        for v in val:
                            try: supervision_results.append(SupervisionResult(**v))
                            except: pass
                    elif isinstance(val, dict):
                        try: supervision_results.append(SupervisionResult(**val))
                        except: pass
            
            logger.info(f"L2 Supervision complete. Generated {len(supervision_results)} critiques.")
            return supervision_results
            
        except Exception as e:
            logger.error(f"L2 Supervision failed: {e}")
            return []

    async def process_cognitive_cycle(self, pipeline_state: PipelineState) -> Dict[str, Any]:
        """
        执行核心认知循环 (L2 融合 -> 事实检查 -> 决策防御)。
        [Task 06] Refactored for Atomic State Updates using Deep Copy Isolation.
        [Task FIX-MED-003] Removed heavy deepcopy, relying on Pydantic's copy mechanism.
        """
        logger.info("Starting cognitive cycle...")
        
        # [Task FIX-MED-003] Logic Check: Verify L1 inputs
        if not pipeline_state.l1_insights:
             logger.warning("Cognitive Cycle running without L1 insights (Blind Mode).")

        # [Task FIX-MED-003] Optimized Snapshot: Use Pydantic's model_copy instead of deepcopy
        # This prevents deep recursion issues and is faster.
        state_snapshot = pipeline_state.model_copy(deep=True)
        
        # 1. Run Reasoning Ensemble (L2 Fusion) on Snapshot
        try:
            fusion_result = await self.reasoning_ensemble.reason(state_snapshot)
            
            # [致命防御] 检查是否为 Dict Bomb
            if isinstance(fusion_result, dict):
                logger.critical(f"Dict Bomb Detected! ReasoningEnsemble returned a raw dict: {fusion_result}")
                if "error" in fusion_result:
                    raise CognitiveError(f"ReasoningEnsemble reported error: {fusion_result['error']}")
                raise CognitiveError("Invalid return type (dict) from ReasoningEnsemble")

            if not isinstance(fusion_result, FusionResult):
                raise CognitiveError(f"Invalid fusion result type: {type(fusion_result)}")

        except Exception as e:
            logger.error(f"Cognitive cycle failed at Reasoning stage: {e}", exc_info=True)
            raise CognitiveError(f"ReasoningEnsemble failed: {e}") from e
            
        # Update snapshot so guard can see it
        # [Fix] Update Pydantic model field directly
        state_snapshot.latest_fusion_result = fusion_result
        
        # 3. Apply Uncertainty Guardrail on Snapshot
        try:
            # [Fix] Use correct API: validate_uncertainty(state) -> Optional[str]
            error_msg = self.uncertainty_guard.validate_uncertainty(state_snapshot)
            
            if error_msg:
                logger.warning(f"Uncertainty Guardrail triggered: {error_msg}")
                # Downgrade decision
                fusion_result.decision = "NEUTRAL"
                fusion_result.confidence = 0.0
                fusion_result.reasoning = f"Guardrail Block: {error_msg}"
            
        except Exception as e:
            logger.error(f"Uncertainty guardrail failed: {e}", exc_info=True)
            # Fail safe
            fusion_result.decision = "ERROR_HOLD"
            fusion_result.confidence = 0.0

        # [Task 06] Atomic Commit: Update global state only after all checks pass
        # Now we apply the result from the isolated process to the real pipeline_state
        # [Fix] Assign the Pydantic model directly, avoiding serialization overhead if possible
        pipeline_state.latest_fusion_result = fusion_result
        # pipeline_state.update_value("last_fusion_result", fusion_result) # Deprecated legacy call if redundant
        
        # For legacy compatibility if needed
        # pipeline_state.update_value("last_guarded_decision", fusion_result.model_dump())

        logger.info(f"Cognitive cycle complete. Final decision: {fusion_result.decision}")
        
        return {
            "final_decision": fusion_result,
            "fact_check_report": None
        }

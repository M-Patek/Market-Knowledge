# Phoenix_project/ai/reasoning_ensemble.py
# [主人喵的修复] 净化了投毒逻辑，强制返回 FusionResult 对象，增加了数据拆包和兜底机制

import logging
import asyncio
from typing import Any, Dict, List, Optional

from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.core.schemas.fusion_result import FusionResult
from Phoenix_project.core.schemas.evidence_schema import EvidenceItem
from Phoenix_project.core.exceptions import CognitiveError
# [Task 1.1 Fix] Import get_logger to prevent NameError
from Phoenix_project.monitor.logging import get_logger

# 依赖组件接口
from Phoenix_project.agents.l2.fusion_agent import FusionAgent
from Phoenix_project.agents.l3.alpha_agent import AlphaAgent
from Phoenix_project.evaluation.voter import Voter
from Phoenix_project.evaluation.arbitrator import Arbitrator
from Phoenix_project.evaluation.fact_checker import FactChecker
from Phoenix_project.data_manager import DataManager

logger = get_logger(__name__)

class ReasoningEnsemble:
    """
    Reasoning Ensemble (L2/L3) Coordination Logic
    负责协调 Fusion, Fact-Checking, Arbitration 和 Alpha Decision。
    [Refactor Fix] 强制类型安全，杜绝 Dict Bomb。
    """

    def __init__(
        self, 
        fusion_agent: FusionAgent, 
        alpha_agent: AlphaAgent,
        voter: Voter, 
        arbitrator: Arbitrator, 
        fact_checker: FactChecker,
        data_manager: DataManager
    ):
        self.fusion_agent = fusion_agent
        self.alpha_agent = alpha_agent
        self.voter = voter
        self.arbitrator = arbitrator
        self.fact_checker = fact_checker
        self.data_manager = data_manager
        logger.info("ReasoningEnsemble initialized with strict type safety.")

    async def reason(self, state: PipelineState) -> FusionResult:
        """
        执行核心推理流程：L1 Insights -> L2 Fusion -> Fact Check -> Decision。
        
        Returns:
            FusionResult: 无论成功与否，必须返回此对象。
        """
        target_symbol = state.main_task_query.get("symbol", "UNKNOWN")
        
        try:
            # 1. 准备依赖数据 (L1 Insights)
            # [Task 1.1 Fix] 拆包逻辑适配 List[EvidenceItem]
            raw_insights = state.l1_insights
            dependencies = self._unwrap_dependencies(raw_insights)
            
            if not dependencies:
                logger.warning("No valid L1 insights found. Returning fallback.")
                return self._create_fallback_result(state, target_symbol, "No L1 insights available.")

            # 2. L2 Fusion (融合)
            logger.info(f"Running L2 Fusion on {len(dependencies)} items...")
            
            # [Task 1.3 Fix] Standardized Sync: await direct return, no generator
            fusion_result = await self.fusion_agent.run(state=state, dependencies=dependencies)

            if not fusion_result:
                logger.error("L2 Fusion Agent yielded no results.")
                return self._create_fallback_result(state, target_symbol, "Fusion Agent yielded empty result.")

            # [双重保险] 再次检查类型，防止 FusionAgent 本身被篡改
            if not isinstance(fusion_result, FusionResult):
                logger.error(f"FusionAgent returned invalid type: {type(fusion_result)}")
                return self._create_fallback_result(state, target_symbol, "Invalid FusionResult type.")

            # 3. Fact Checking (事实核查) - 简化版，Engine 可能会再次调用
            # 这里我们可以做一些预处理
            if fusion_result.confidence > 0.6:
                await self._apply_fact_check(fusion_result)

            # 4. Arbitration (仲裁) - 可选
            # 如果有严重冲突，可以在这里调用 Arbitrator
            # ...

            return fusion_result

        except Exception as e:
            logger.error(f"ReasoningEnsemble critical failure: {e}", exc_info=True)
            # [最终兜底] 无论发生什么，都返回一个安全的对象
            return self._create_fallback_result(state, target_symbol, f"Critical Ensemble Error: {str(e)}")

    def _unwrap_dependencies(self, raw_insights: List[EvidenceItem]) -> List[EvidenceItem]:
        """
        [Task 1.1 Fix] 清洗逻辑优化：直接处理 List[EvidenceItem]
        """
        if not raw_insights:
            return []
        
        # Filter and ensure type safety
        return [item for item in raw_insights if isinstance(item, EvidenceItem)]

    async def _apply_fact_check(self, fusion_result: FusionResult):
        """
        辅助方法：运行事实核查并原地更新 FusionResult。
        """
        try:
            claims = [fusion_result.reasoning]
            results = await self.fact_checker.check_facts(claims)
            
            if not results:
                return

            # [Fix] Handle List[FactCheckResult] return type
            main_report = results[0]
            is_verified = getattr(main_report, "verified", False)

            if not is_verified:
                fusion_result.decision = "NEUTRAL"
                fusion_result.confidence = 0.0
                fusion_result.reasoning += "\n[FACT-CHECK]: Reasoning Unverified/Refuted. Decision neutralized."
            else:
                # [Task 1.2 Fix] Removed magic number logic (+0.1) - Purity
                pass
                
        except Exception as e:
            logger.warning(f"Fact check failed (non-fatal): {e}")

    def _create_fallback_result(self, state: PipelineState, symbol: str, reason: str) -> FusionResult:
        """
        生成标准化的兜底结果。
        """
        return FusionResult(
            timestamp=state.current_time,
            target_symbol=symbol,
            decision="NEUTRAL", # 保持中立
            confidence=0.0,     # 零置信度
            reasoning=f"Ensemble Fallback Triggered: {reason}",
            uncertainty=1.0,    # 最大不确定性
            supporting_evidence_ids=[],
            conflicting_evidence_ids=[]
        )

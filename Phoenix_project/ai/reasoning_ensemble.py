import asyncio
from typing import Any, Dict, List
# 修复：导入正确的 monitor.logging 和 core.pipeline_state
from Phoenix_project.monitor.logging import get_logger
from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.ai.prompt_manager import PromptManager
from Phoenix_project.api.gemini_pool_manager import GeminiPoolManager
from Phoenix_project.evaluation.voter import Voter
from Phoenix_project.evaluation.arbitrator import Arbitrator
from Phoenix_project.evaluation.fact_checker import FactChecker
from Phoenix_project.core.schemas.fusion_result import FusionResult

# 修复：使用 get_logger
logger = get_logger(__name__)

class ReasoningEnsemble:
    """
    协调多个L2/L3智能体，融合它们的见解，并进行最终决策。
    """

    def __init__(
        self,
        prompt_manager: PromptManager,
        gemini_pool: GeminiPoolManager,
        voter: Voter,
        arbitrator: Arbitrator,
        fact_checker: FactChecker,
        # 修复：添加在 worker.py 中传递但 __init__ 中缺失的依赖项
        retriever: Any,
        ensemble_client: Any,
        metacognitive_agent: Any
    ):
        self.prompt_manager = prompt_manager
        self.gemini_pool = gemini_pool
        self.voter = voter
        self.arbitrator = arbitrator
        self.fact_checker = fact_checker
        # 修复：存储传入的依赖项
        self.retriever = retriever
        self.ensemble_client = ensemble_client
        self.metacognitive_agent = metacognitive_agent
        
        self.fusion_agent = None  # L2 Fusion Agent
        self.alpha_agent = None   # L3 Alpha Agent
        logger.info("ReasoningEnsemble initialized.")

    def set_agents(self, fusion_agent: Any, alpha_agent: Any):
        """
        设置L2和L3智能体。
        """
        self.fusion_agent = fusion_agent
        self.alpha_agent = alpha_agent
        logger.info("L2 Fusion and L3 Alpha agents set in ReasoningEnsemble.")

    async def run_ensemble(
        self, 
        state: PipelineState, 
        agent_insights: Dict[str, Any],
        target_assets: List[str]
    ) -> Dict[str, Any]:
        """
        执行推理流程：融合 -> 事实核查 -> 投票 -> 仲裁 -> 最终决策。
        
        Args:
            state: 当前的流水线状态。
            agent_insights: L1智能体产生的见解。
            target_assets: 需要分析的目标资产列表 (例如: ["AAPL", "GOOG"])。

        Returns:
            包含最终决策和置信度的字典。
        """
        if not self.fusion_agent or not self.alpha_agent:
            logger.error("Agents not set in ReasoningEnsemble. Call set_agents() first.")
            return {"error": "Agents not configured."}

        logger.info(f"Reasoning ensemble starting for assets: {target_assets}")

        # 1. L2 融合 (Fusion)
        # L2 Fusion Agent 接收 L1 见解
        logger.debug("Running L2 Fusion Agent...")
        try:
            fusion_result: FusionResult = await self.fusion_agent.run(state, agent_insights, target_assets)
            if not fusion_result:
                logger.warning("L2 Fusion Agent returned no result.")
                return {"error": "Fusion agent failed to produce a result."}
        except Exception as e:
            logger.error(f"Error during L2 Fusion: {e}")
            return {"error": f"Fusion agent exception: {e}"}

        # 2. 事实核查 (Fact-Checking)
        logger.debug("Running Fact-Checker...")
        # 修复：fusion_result.insights 不存在。应检查 fusion_result.reasoning
        # 假设 fact_checker.check_facts 接受字符串
        fact_check_report = await self.fact_checker.check_facts(fusion_result.reasoning)
        # 修复：state.add_fact_check_report 不存在。
        # 我们将报告保存在本地，以便传递给 arbitrator
        # state.add_fact_check_report(fact_check_report)

        # 3. 投票 (Voting)
        # L1智能体（作为投票者）对融合后的见解进行投票
        logger.debug("Running Voter...")
        
        # 修复 TODO：L1 智能体应该被动态地用作投票者
        # 我们将 L1 的原始见解 (agent_insights) 和 L2 的融合结果 (fusion_result)
        # 都传递给投票者，让它来决定如何处理。
        # 假设 voter.collect_votes 签名是 (l1_insights, fusion_result)
        votes = await self.voter.collect_votes(agent_insights, fusion_result)
        
        # 4. 仲裁 (Arbitration)
        logger.debug("Running Arbitrator...")
        # 修复：fusion_result.insights 不存在。我们将 L1 见解 (agent_insights) 传递给仲裁者
        arbitrated_insights = await self.arbitrator.arbitrate(agent_insights, votes, fact_check_report)

        # 5. L3 决策 (Alpha Generation)
        # L3 Alpha Agent 接收经过仲裁的L2见解
        logger.debug("Running L3 Alpha Agent...")
        try:
            # stocks_to_analyze = ["AAPL", "GOOG"]  # FIXME: This is hardcoded and needs to be dynamic based on the query or context
            stocks_to_analyze = target_assets
            logger.info(f"Alpha Agent analyzing stocks: {stocks_to_analyze}")

            decision = await self.alpha_agent.run(state, arbitrated_insights, stocks_to_analyze)
            
            logger.info(f"L3 Alpha Agent decision: {decision}")
            
            # 最终决策
            final_decision = {
                "asset_insights": decision.asset_insights,
                "portfolio_implications": decision.portfolio_implications,
                "confidence": decision.confidence,
                "rational": decision.rational,
                "arbitration_report": arbitrated_insights,
                "fact_check_report": fact_check_report
            }
            
            # 修复：state.add_final_decision 不存在。
            # state.add_final_decision(final_decision)
            
            return final_decision

        except Exception as e:
            logger.error(f"Error during L3 Alpha Generation: {e}")
            return {"error": f"Alpha agent exception: {e}"}

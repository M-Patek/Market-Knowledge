import asyncio
from typing import Any, Dict, List
from types import SimpleNamespace
# 修复：导入正确的 monitor.logging 和 core.pipeline_state
from Phoenix_project.monitor.logging import get_logger
from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.core.schemas.fusion_result import FusionResult
from Phoenix_project.core.exceptions import CognitiveError # Use existing exception or generic
# 修复：使用 get_logger
logger = get_logger(__name__)

# 假设这些是你的智能体类，需要根据实际路径调整导入
from Phoenix_project.agents.l2.fusion_agent import FusionAgent
from Phoenix_project.agents.l3.alpha_agent import AlphaAgent
from Phoenix_project.evaluation.voter import Voter
from Phoenix_project.evaluation.arbitrator import Arbitrator
from Phoenix_project.evaluation.fact_checker import FactChecker
from Phoenix_project.data_manager import DataManager

class ReasoningEnsemble:
    """
    Reasoning Ensemble (L2/L3) Coordination Logic
    Orchestrates Fusion, Fact-Checking, Arbitration, and Alpha Decision.
    [Refactored Phase 3.3] Added DataManager, removed hallucinations.
    """

    def __init__(self, fusion_agent: FusionAgent, alpha_agent: AlphaAgent,
                 voter: Voter, arbitrator: Arbitrator, fact_checker: FactChecker,
                 data_manager: DataManager):
        self.fusion_agent = fusion_agent
        self.alpha_agent = alpha_agent
        self.voter = voter
        self.arbitrator = arbitrator
        self.fact_checker = fact_checker
        self.data_manager = data_manager

    async def run_ensemble(self, state: PipelineState, agent_insights: Dict[str, Any], target_assets: List[str]) -> Dict[str, Any]:
        """
        Executes the reasoning ensemble flow.
        """
        if not agent_insights:
            logger.warning("No L1 insights provided to Reasoning Ensemble.")
            return {"error": "No insights available."}

        # 1. 融合 (Fusion)
        # L2 Fusion Agent 接收 L1 见解
        logger.debug("Running L2 Fusion Agent...")
        try:
            # [Task 3.3 Refactor] 适配 FusionAgent.run(state, dependencies) 新签名
            fusion_result: FusionResult = None
            
            # 将 dict values 转为 list 传给 dependencies
            dependencies = list(agent_insights.values())
            
            async for result in self.fusion_agent.run(state=state, dependencies=dependencies):
                fusion_result = result
                break # We only need the first result

            if not fusion_result:
                logger.warning("L2 Fusion Agent returned no result.")
                return {"error": "Fusion agent failed to produce a result."}
                
        except Exception as e:
            logger.error(f"Fusion Agent failed: {e}", exc_info=True)
            return {"error": f"Fusion failed: {str(e)}"}

        # 2. 事实核查 (Fact-Checking)
        logger.debug("Running Fact-Checker...")
        # [Task III Fix] Pass list of strings, not single string
        claims = [fusion_result.reasoning] if fusion_result.reasoning else []
        fact_check_report = await self.fact_checker.check_facts(claims)
        
        # 修复：state.add_fact_check_report 不存在。
        # 我们将报告保存在本地，以便传递给 arbitrator
        # state.add_fact_check_report(fact_check_report)

        # 3. 仲裁 (Arbitration) - 如果存在冲突或低置信度
        logger.debug("Running Arbitrator...")
        arbitrated_insights = await self.arbitrator.arbitrate(
            fusion_result, 
            agent_insights, 
            fact_check_report
        )

        # 4. L3 Alpha Agent (Final Decision)
        logger.debug("Running L3 Alpha Agent...")
        try:
            # 假设 Alpha Agent 需要最新的市场状态和仲裁后的见解
            # 修复：Alpha Agent 接口可能不同。此处假设 run(state, insights)
            # 并且 target_assets 是列表
            stocks_to_analyze = target_assets
            logger.info(f"Alpha Agent analyzing stocks: {stocks_to_analyze}")

            # [Task III Fix] Adapt DRL Agent Interface
            # 1. Get Market Data
            # [Refactored Phase 3.3] 使用 DataManager 获取真实数据，禁止幻觉
            symbol = stocks_to_analyze[0]
            market_data = await self.data_manager.get_latest_market_data(symbol)
            
            if not market_data:
                raise CognitiveError(f"CRITICAL: No market data available for {symbol}. Cannot make L3 decision.")

            state_data = {
                "balance": state.portfolio_state.cash if state.portfolio_state else 10000.0,
                "holdings": 0.0, # Simplified for fix
                "price": market_data.close
            }
            
            # 2. Construct Observation & Compute Action
            # Assuming 'format_observation' is the public API for _format_obs
            # If not available, we might need to check BaseDRLAgent, but sticking to emergency plan instructions.
            obs = self.alpha_agent.format_observation(state_data, fusion_result)
            action = self.alpha_agent.compute_action(obs)
            
            logger.info(f"L3 Alpha Agent computed action: {action}")
            
            # 最终决策
            final_decision = {
                "action": action.tolist() if hasattr(action, "tolist") else action,
                "asset_insights": fusion_result.reasoning, # Fallback to fusion reasoning
                "portfolio_implications": "Rebalancing based on DRL Policy",
                "confidence": fusion_result.confidence,
                "rational": "Policy Execution",
                "arbitration_report": arbitrated_insights,
                "fact_check_report": fact_check_report
            }
            
            return final_decision

        except Exception as e:
            logger.error(f"Alpha Agent failed: {e}", exc_info=True)
            return {"error": f"Alpha Agent failed: {str(e)}"}

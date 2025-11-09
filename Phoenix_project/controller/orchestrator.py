import asyncio
from typing import Dict, Any, List
from monitor.logging import logger
from core.pipeline_state import PipelineState
from core.schemas.task_schema import Task
from ai.reasoning_ensemble import ReasoningEnsemble
from agents.executor import AgentExecutor
from data_manager import DataManager
from context_bus import ContextBus

class Orchestrator:
    """
    协调数据流、智能体执行和推理。
    管理整个认知架构的端到端执行循环。
    """
    def __init__(
        self,
        state: PipelineState,
        agent_executor: AgentExecutor,
        reasoning_ensemble: ReasoningEnsemble,
        data_manager: DataManager,
        context_bus: ContextBus
    ):
        self.state = state
        self.agent_executor = agent_executor
        self.reasoning_ensemble = reasoning_ensemble
        self.data_manager = data_manager
        self.context_bus = context_bus
        logger.info("Orchestrator initialized.")

    async def process_task(self, task: Task) -> Dict[str, Any]:
        """
        处理单个任务（例如，用户查询）。
        """
        logger.info(f"Processing task: {task.task_id} - {task.description}")
        
        # 1. 更新状态
        self.state.set_current_task(task)
        
        # 2. 触发数据管理器
        # Orchestrator 告诉 DataManager 需要什么数据
        # FIXME: 从任务中提取目标资产
        # target_assets = self.extract_assets_from_task(task)
        # 临时硬编码
        target_assets = ["AAPL", "GOOG"] 
        
        try:
            market_data = await self.data_manager.fetch_market_data(target_assets)
            news_data = await self.data_manager.fetch_news_data(target_assets)
            
            # 更新状态
            self.state.add_market_data(market_data)
            self.state.add_news_data(news_data)
            
            logger.info("Market data and news data fetched and added to state.")

        except Exception as e:
            logger.error(f"Failed to fetch data: {e}")
            return {"error": f"Data fetching failed: {e}"}

        # 3. 运行分析层
        try:
            final_decision = await self.run_analysis_level(self.state)
            logger.info(f"Task {task.task_id} processed. Final decision: {final_decision.get('asset_insights')}")
            return final_decision
        except Exception as e:
            logger.error(f"Error during analysis level execution: {e}")
            return {"error": f"Analysis level failed: {e}"}

    async def run_analysis_level(self, state: PipelineState) -> Dict[str, Any]:
        """
        执行完整的 L1 -> L2 -> L3 分析流程。
        """
        logger.info("Running analysis level...")

        # 从状态中获取目标资产
        market_data = state.get_latest_market_data()
        if not market_data:
            logger.warning("No market data found in state. Analysis level cannot proceed.")
            return {"error": "Missing market data."}
            
        # 从市场数据中提取资产（例如，["AAPL", "GOOG"]）
        target_assets = list(market_data.keys())
        if not target_assets:
            logger.warning("Market data is empty. No target assets to analyze.")
            return {"error": "No target assets found in market data."}
            
        logger.info(f"Analysis level targeting assets: {target_assets}")


        # 1. 运行 L1 智能体
        logger.debug("Executing L1 agents...")
        try:
            # L1 智能体并行运行
            l1_insights = await self.agent_executor.run_l1_agents(state, target_assets)
            if not l1_insights:
                logger.warning("L1 agents produced no insights.")
                return {"error": "L1 agents failed to produce insights."}
            logger.info(f"L1 agents completed. {len(l1_insights)} insights generated.")
        except Exception as e:
            logger.error(f"Error executing L1 agents: {e}", exc_info=True)
            return {"error": f"L1 agent execution failed: {e}"}

        # 2. 运行 L2 智能体 (监督层)
        logger.debug("Executing L2 agents...")
        try:
            # L2 智能体并行运行
            l2_supervision = await self.agent_executor.run_l2_agents(state, l1_insights)
            logger.info("L2 agents completed.")
        except Exception as e:
            logger.error(f"Error executing L2 agents: {e}", exc_info=True)
            return {"error": f"L2 agent execution failed: {e}"}
        
        # TODO: L2 见解 (l2_supervision) 应该被传递给推理集合
        # 目前，推理集合 (ReasoningEnsemble) 内部有自己的L2/L3智能体
        
        # 3. 运行推理集合 (L2 Fusion -> L3 Alpha)
        logger.debug("Executing Reasoning Ensemble...")
        try:
            # ReasoningEnsemble 协调 L2 Fusion 和 L3 Alpha 智能体
            final_insights = await self.reasoning_ensemble.run_ensemble(state, l1_insights, target_assets)
            logger.info("Reasoning Ensemble completed.")
            return final_insights
        except Exception as e:
            logger.error(f"Error executing Reasoning Ensemble: {e}", exc_info=True)
            return {"error": f"Reasoning Ensemble failed: {e}"}

    async def run_system_tick(self):
        """
        由 LoopManager 调用的常规系统“心跳”。
        用于后台任务、数据摄取和监控。
        """
        logger.info("Orchestrator system tick running...")
        
        # 1. 触发数据管理器进行后台刷新
        try:
            await self.data_manager.refresh_data_sources()
            logger.debug("Data sources refreshed.")
        except Exception as e:
            logger.warning(f"Failed to refresh data sources during system tick: {e}")

        # 2. 运行监控/内省智能体 (如果需要)
        # (例如, L2 MetacognitiveAgent)
        try:
            l2_meta_agent = self.agent_executor.get_agent("L2_MetacognitiveAgent")
            if l2_meta_agent:
                logger.debug("Running L2 MetacognitiveAgent on system tick...")
                # 元认知智能体检查整个状态
                await l2_meta_agent.run(self.state) 
            
        except Exception as e:
            logger.warning(f"Failed to run metacognitive agent during system tick: {e}")

        # 3. 向上下文总线广播系统状态更新
        system_health = {"status": "OK", "timestamp": self.state.get_current_time()}
        self.context_bus.publish("system_health", system_health)
        
        logger.info("Orchestrator system tick completed.")

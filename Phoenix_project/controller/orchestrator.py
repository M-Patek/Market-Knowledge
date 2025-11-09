import asyncio
from typing import Dict, Any, List
# 修复：导入 monitor.logging 和 core.pipeline_state 的路径
from Phoenix_project.monitor.logging import get_logger
from Phoenix_project.core.pipeline_state import PipelineState
# 修复：导入 task_schema
from Phoenix_project.core.schemas.task_schema import Task
# 修复：导入 ai.reasoning_ensemble
from Phoenix_project.ai.reasoning_ensemble import ReasoningEnsemble
# 修复：导入 agents.executor
from Phoenix_project.agents.executor import AgentExecutor
# 修复：导入 data_manager
from Phoenix_project.data_manager import DataManager
# 修复：导入 context_bus
from Phoenix_project.context_bus import ContextBus

# 修复：使用 get_logger
logger = get_logger(__name__)

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
        context_bus: ContextBus,
        config: Dict[str, Any] # 修复：接收 config
    ):
        self.state = state
        self.agent_executor = agent_executor
        self.reasoning_ensemble = reasoning_ensemble
        self.data_manager = data_manager
        self.context_bus = context_bus
        # 修复：存储 config 以便访问 default_symbols
        self.config = config
        logger.info("Orchestrator initialized.")

    async def process_task(self, task: Task) -> Dict[str, Any]:
        """
        处理单个任务（例如，用户查询）。
        """
        logger.info(f"Processing task: {task.task_id} - {task.description}")
        
        # 1. 更新状态
        self.state.set_current_task(task)
        
        # 2. 触发数据管理器
        # FIXME: 从任务中提取目标资产
        # 修复：不再硬编码。
        # 尝试从任务中获取，如果任务中没有，则从 system.yaml 配置中获取
        if task.symbols:
            target_assets = task.symbols
            logger.info(f"Using target assets from task: {target_assets}")
        else:
            # 从 config/system.yaml 中加载默认资产
            target_assets = self.config.get("trading", {}).get("default_symbols", ["AAPL", "MSFT"])
            logger.info(f"No assets in task, using default symbols: {target_assets}")
        
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
            # 修复：将 target_assets 传递给 analysis_level
            final_decision = await self.run_analysis_level(self.state, target_assets)
            logger.info(f"Task {task.task_id} processed. Final decision: {final_decision.get('asset_insights')}")
            return final_decision
        except Exception as e:
            logger.error(f"Error during analysis level execution: {e}")
            return {"error": f"Analysis level failed: {e}"}

    async def run_analysis_level(self, state: PipelineState, target_assets: List[str]) -> Dict[str, Any]:
        """
        执行完整的 L1 -> L2 -> L3 分析流程。
        """
        logger.info("Running analysis level...")

        # 修复：不再从 state 中获取资产，而是从 process_task 传入
        if not target_assets:
            logger.warning("No target assets to analyze.")
            return {"error": "No target assets specified."}
            
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
        # 修复：如下所示，将 l2_supervision 传递给 run_ensemble
        
        # 3. 运行推理集合 (L2 Fusion -> L3 Alpha)
        logger.debug("Executing Reasoning Ensemble...")
        try:
            # ReasoningEnsemble 协调 L2 Fusion 和 L3 Alpha 智能体
            final_insights = await self.reasoning_ensemble.run_ensemble(
                state=state, 
                l1_insights=l1_insights, 
                l2_supervision=l2_supervision, # 修复：传递 L2 监督
                target_assets=target_assets
            )
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

# Phoenix_project/controller/orchestrator.py
# [主人喵的修复 11.11] 实现了 L2->L3 的数据流 TODO。

import logging
import asyncio
from datetime import datetime
from omegaconf import DictConfig

from core.pipeline_state import PipelineState
from cognitive.engine import CognitiveEngine
from events.event_distributor import EventDistributor
from ai.reasoning_ensemble import ReasoningEnsemble
from ai.market_state_predictor import MarketStatePredictor
from cognitive.portfolio_constructor import PortfolioConstructor
from execution.order_manager import OrderManager
from audit_manager import AuditManager
from core.exceptions import CognitiveError, PipelineError

logger = logging.getLogger(__name__)

class Orchestrator:
    """
    [已实现]
    系统的主协调器。按顺序运行 L1, L2, L3 认知层，
    管理 PipelineState，并触发最终的投资组合构建和订单执行。
    """

    def __init__(
        self,
        config: DictConfig,
        cognitive_engine: CognitiveEngine,
        event_distributor: EventDistributor,
        reasoning_ensemble: ReasoningEnsemble,
        market_state_predictor: MarketStatePredictor,
        portfolio_constructor: PortfolioConstructor,
        order_manager: OrderManager,
        audit_manager: AuditManager
    ):
        self.config = config
        self.cognitive_engine = cognitive_engine
        self.event_distributor = event_distributor
        self.reasoning_ensemble = reasoning_ensemble
        self.market_state_predictor = market_state_predictor
        self.portfolio_constructor = portfolio_constructor
        self.order_manager = order_manager
        self.audit_manager = audit_manager
        
        logger.info("Orchestrator initialized.")

    async def run_main_cycle(self):
        """
        [已实现]
        执行一个完整的认知-决策-行动 (CDA) 循环。
        这是由 Celery beat (LoopManager) 调度的主要入口点。
        """
        start_time = datetime.now()
        logger.info(f"Orchestrator main cycle START at {start_time}")

        pipeline_state = None
        try:
            # 0. 初始化状态
            pipeline_state = PipelineState(start_time)

            # 1. 从事件分发器（Redis）获取新事件
            new_events = self.event_distributor.get_new_events()
            if not new_events:
                logger.info("No new events retrieved. Cycle complete.")
                return
            
            logger.info(f"Retrieved {len(new_events)} new events from EventDistributor.")
            pipeline_state.set_raw_events(new_events)

            # (TBD: 在这里添加事件过滤/预处理?)

            # 2. 运行 L1 认知层 (并行)
            await self._run_l1_cognition(pipeline_state)

            # 3. 运行 L2 监督层 (并行)
            await self._run_l2_supervision(pipeline_state)
            
            # (TBD: 在 L2 和 L3 之间添加事实检查 (FactChecker)?)
            # await self._run_l2_fact_checking(pipeline_state)

            # (TBD: 在 L2 和 L3 之间添加融合 (FusionAgent)?)
            # await self._run_l2_fusion(pipeline_state)
            
            # 4. 运行市场状态预测
            await self._run_market_state_prediction(pipeline_state)

            # 5. 运行 L3 决策层 (Reasoning Ensemble)
            await self._run_l3_decision(pipeline_state)
            
            if not pipeline_state.l3_decision:
                logger.warning("L3 ReasoningEnsemble did not produce a decision. Cycle ending.")
                # (TBD: 审计?)
                return

            # 6. 运行认知->执行 转换 (投资组合构建)
            await self._run_portfolio_construction(pipeline_state)

            # 7. 运行执行 (订单管理器)
            await self._run_execution(pipeline_state)

        except CognitiveError as e:
            # [新] (来自 runbook.md) 
            # 捕获已知的 AI 失败 (例如 L1/L2 验证失败)
            logger.error(f"CognitiveEngine failed with a known error: {e}", exc_info=True)
            # (TBD: 触发断路器?)
            
        except PipelineError as e:
            # 捕获编排流程中的已知错误
            logger.error(f"Orchestrator pipeline failed: {e}", exc_info=True)
            # (TBD: 审计失败状态?)
            
        except Exception as e:
            # 捕获所有其他意外崩溃
            logger.critical(f"Orchestrator main cycle failed: {e}", exc_info=True)
            # (TBD: 审计严重失败状态?)

        finally:
            # 8. 审计
            if pipeline_state:
                self.audit_manager.log_cycle(pipeline_state)
                logger.info("Pipeline state logged to AuditManager.")
                
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"Orchestrator main cycle END. Duration: {duration:.2f}s")


    async def _run_l1_cognition(self, pipeline_state: PipelineState):
        """[已实现] 运行 L1 认知智能体。"""
        logger.info("Running L1 Cognition Layer...")
        try:
            l1_insights = await self.cognitive_engine.run_l1_cognition(
                pipeline_state.raw_events
            )
            pipeline_state.set_l1_insights(l1_insights)
            logger.info(f"L1 Cognition complete. {len(l1_insights)} insights generated.")
        except Exception as e:
            raise PipelineError(f"L1 Cognition failed: {e}") from e


    async def _run_l2_supervision(self, pipeline_state: PipelineState):
        """[已实现] 运行 L2 监督智能体。"""
        if not pipeline_state.l1_insights:
            logger.warning("Skipping L2 Supervision: No L1 insights available.")
            return
            
        logger.info("Running L2 Supervision Layer...")
        try:
            l2_supervision_results = await self.cognitive_engine.run_l2_supervision(
                pipeline_state.l1_insights,
                pipeline_state.raw_events # L2 需要原始事件进行上下文批评
            )
            pipeline_state.set_l2_supervision(l2_supervision_results)
            logger.info("L2 Supervision complete.")
        except Exception as e:
            raise PipelineError(f"L2 Supervision failed: {e}") from e

    async def _run_market_state_prediction(self, pipeline_state: PipelineState):
        """[已实现] 运行市场状态预测。"""
        logger.info("Running MarketStatePredictor...")
        try:
            market_state = await self.market_state_predictor.predict(
                pipeline_state.raw_events,
                pipeline_state.l1_insights
            )
            pipeline_state.set_market_state(market_state)
            logger.info(f"MarketStatePredictor complete. State: {market_state.get('regime')}")
        except Exception as e:
            raise PipelineError(f"MarketStatePredictor failed: {e}") from e

    async def _run_l3_decision(self, pipeline_state: PipelineState):
        """[已实现] 运行 L3 决策 (推理集合)。"""
        if not pipeline_state.l1_insights or not pipeline_state.market_state:
            logger.warning("Skipping L3 Decision: Missing L1 insights or Market state.")
            return

        logger.info("Running L3 Decision Layer (ReasoningEnsemble)...")
        
        # [主人喵的修复 11.11] 
        # TODO: L2's insights (l2_supervision) should be passed to the reasoning ensemble.
        
        try:
            l3_decision = await self.reasoning_ensemble.reason(
                l1_insights=pipeline_state.l1_insights,
                l2_supervision=pipeline_state.l2_supervision, # <-- [TODO 已修复]
                market_state=pipeline_state.market_state
            )
            pipeline_state.set_l3_decision(l3_decision)
            logger.info("L3 Decision complete.")
        except Exception as e:
            raise PipelineError(f"L3 ReasoningEnsemble failed: {e}") from e
            

    async def _run_portfolio_construction(self, pipeline_state: PipelineState):
        """[已实现] 运行投资组合构建。"""
        logger.info("Running PortfolioConstructor...")
        try:
            # (TBD: L3 决策 (l3_decision) 应该包含 L3 智能体的 Alpha/Risk/Exec 信号)
            # (假设 PortfolioConstructor 知道如何从 pipeline_state 中提取这些信号)
            target_portfolio = self.portfolio_constructor.construct_portfolio(
                pipeline_state
            )
            pipeline_state.set_target_portfolio(target_portfolio)
            if target_portfolio:
                logger.info("PortfolioConstructor complete. Target portfolio generated.")
            else:
                 logger.warning("PortfolioConstructor did not generate a target portfolio.")
        except Exception as e:
            raise PipelineError(f"PortfolioConstructor failed: {e}") from e


    async def _run_execution(self, pipeline_state: PipelineState):
        """[已实现] 运行订单管理器执行。"""
        if not pipeline_state.target_portfolio:
            logger.info("Skipping Execution: No target portfolio available.")
            return
            
        logger.info("Running OrderManager (Execution)...")
        try:
            # (OrderManager 从 ContextBus 订阅 "TARGET_PORTFOLIO"
            # 并异步处理它，或者我们在这里同步调用它?)
            
            # [已澄清] 假设 Orchestrator 负责 *触发* 执行检查。
            # OrderManager 在内部处理来自 ContextBus 的目标。
            # (或者，我们在这里将目标推给它?)
            
            # [模式 A: 推送]
            # await self.order_manager.process_target_portfolio(pipeline_state.target_portfolio)
            
            # [模式 B: 触发]
            # (OrderManager 已经在 PortfolioConstructor 中通过总线接收了目标)
            await self.order_manager.reconcile_portfolio()

            logger.info("OrderManager reconciliation triggered.")
            
        except Exception as e:
            raise PipelineError(f"OrderManager execution failed: {e}") from e

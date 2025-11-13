# Phoenix_project/controller/orchestrator.py
# [主人喵的修复 11.11] 实现了 L2->L3 的数据流 TODO。
# [主人喵的修复 11.12] 实现了 TBD-1 (事件过滤) 和 TBD-5 (审计)

import logging
import asyncio
from datetime import datetime
from omegaconf import DictConfig
from typing import List, Dict, Optional, Any # [Fix II.1]

from core.pipeline_state import PipelineState
from cognitive.engine import CognitiveEngine
from events.event_distributor import EventDistributor
from events.risk_filter import EventRiskFilter # [TBD-1 修复] 导入过滤器
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
        event_filter: EventRiskFilter, # [TBD-1 修复] 注入过滤器
        reasoning_ensemble: ReasoningEnsemble,
        market_state_predictor: MarketStatePredictor,
        portfolio_constructor: PortfolioConstructor,
        order_manager: OrderManager,
        audit_manager: AuditManager,
        # [Fix II.1] 注入 L3 DRL 智能体
        alpha_agent: Optional[Any] = None,
        risk_agent: Optional[Any] = None,
        execution_agent: Optional[Any] = None
        # (TBD-2/3 FactChecker/FusionAgent 由 ReasoningEnsemble 内部处理)
    ):
        self.config = config
        self.cognitive_engine = cognitive_engine
        self.event_distributor = event_distributor
        self.event_filter = event_filter # [TBD-1 修复] 存储过滤器
        self.reasoning_ensemble = reasoning_ensemble
        self.market_state_predictor = market_state_predictor
        self.portfolio_constructor = portfolio_constructor
        self.order_manager = order_manager
        self.audit_manager = audit_manager
        # [Fix II.1] 存储 L3 智能体
        self.alpha_agent = alpha_agent
        self.risk_agent = risk_agent
        self.execution_agent = execution_agent
        
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

            # [TBD-1 修复] 在 L1 认知层之前添加事件过滤/预处理的逻辑。
            logger.info(f"Filtering {len(new_events)} events...")
            filtered_events = self.event_filter.filter_batch(new_events)
            
            if not filtered_events:
                logger.info("No events remaining after filtering. Cycle complete.")
                return
            
            logger.info(f"{len(filtered_events)} events remaining after filtering.")
            # (我们将 'filtered_events' 传递给 L1)

            # 2. 运行 L1 认知层 (并行)
            await self._run_l1_cognition(pipeline_state, filtered_events) # [TBD-1 修复]

            # 3. 运行 L2 监督层 (并行)
            await self._run_l2_supervision(pipeline_state)
            
            # [TBD-2/3 修复]
            # (TBD-2: FactChecker 和 TBD-3: FusionAgent 的逻辑)
            # (根据 ai/reasoning_ensemble.py，这些由 L3 的 ReasoningEnsemble 处理)
            
            # 4. 运行市场状态预测
            await self._run_market_state_prediction(pipeline_state)

            # 5. 运行 L3 决策层 (Reasoning Ensemble)
            # TODO: L3 决策应使用 [Fix II.1] 中注入的 DRL 智能体 (self.alpha_agent 等)
            await self._run_l3_decision(pipeline_state)
            
            if not pipeline_state.l3_decision:
                logger.warning("L3 ReasoningEnsemble did not produce a decision. Cycle ending.")
                # [TBD-5 修复] 审计决策失败
                if pipeline_state:
                    await self.audit_manager.audit_event(
                        event_type="DECISION_FAILURE",
                        details={"reason": "L3 ReasoningEnsemble (or L2 Fusion) did not produce a decision."},
                        pipeline_state=pipeline_state
                    )
                return

            # 6. 运行认知->执行 转换 (投资组合构建)
            await self._run_portfolio_construction(pipeline_state)

            # 7. 运行执行 (订单管理器)
            await self._run_execution(pipeline_state)

        except CognitiveError as e:
            # [新] (来自 runbook.md) 
            # 捕获已知的 AI 失败 (例如 L1/L2 验证失败)
            logger.error(f"CognitiveEngine failed with a known error: {e}", exc_info=True)
            # [TBD-5 修复] 审计
            if pipeline_state:
               await self.audit_manager.audit_error(e, "CognitiveEngine", pipeline_state)
            
        except PipelineError as e:
            # 捕获编排流程中的已知错误
            logger.error(f"Orchestrator pipeline failed: {e}", exc_info=True)
            # [TBD-5 修复] 审计
            if pipeline_state:
               await self.audit_manager.audit_error(e, "OrchestratorPipeline", pipeline_state)
            
        except Exception as e:
            # 捕获所有其他意外崩溃
            logger.critical(f"Orchestrator main cycle failed: {e}", exc_info=True)
            # [TBD-5 修复] 审计
            if pipeline_state:
               await self.audit_manager.audit_error(e, "OrchestratorFatal", pipeline_state)

        finally:
            # 8. 审计
            if pipeline_state:
                self.audit_manager.log_cycle(pipeline_state)
                logger.info("Pipeline state logged to AuditManager.")
                
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"Orchestrator main cycle END. Duration: {duration:.2f}s")


    async def _run_l1_cognition(self, pipeline_state: PipelineState, filtered_events: List[Dict]): # [TBD-1 修复]
        """[已实现] 运行 L1 认知智能体。"""
        logger.info("Running L1 Cognition Layer...")
        try:
            # [TBD-1 修复] 使用过滤后的事件
            l1_insights = await self.cognitive_engine.run_l1_cognition(
                filtered_events
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
        
        logger.info("Running L3 Decision Layer (DRL Agents)...")
        
        try:
            # [Task I.1 & I.2] Replace ReasoningEnsemble with DRL Agents
            
            # 1. Prepare Context / State Data
            # Identify the symbol we are trading (Assuming single-symbol focus for now based on task query)
            task_query = pipeline_state.get_main_task_query()
            symbol = task_query.get("symbol", "BTC/USD") # Default fallback
            
            market_data = pipeline_state.get_latest_market_data(symbol)
            price = market_data.close if market_data else 0.0
            
            pf_state = pipeline_state.portfolio_state
            balance = pf_state.cash if pf_state else 0.0
            # Get holdings for the specific symbol
            holdings = 0.0
            if pf_state and symbol in pf_state.positions:
                holdings = pf_state.positions[symbol].quantity

            state_data = {
                "balance": balance,
                "holdings": holdings,
                "price": price,
                "symbol": symbol
            }

            # 2. Alpha Agent Decision
            alpha_action = None
            if self.alpha_agent:
                obs = self.alpha_agent.format_observation(state_data, pipeline_state.l2_supervision)
                alpha_action = self.alpha_agent.compute_action(obs)
                logger.info(f"Alpha Agent Action: {alpha_action}")

            # 3. Risk Agent Decision (Optional)
            risk_action = None
            if self.risk_agent:
                obs = self.risk_agent.format_observation(state_data, pipeline_state.l2_supervision)
                risk_action = self.risk_agent.compute_action(obs)
                logger.info(f"Risk Agent Action: {risk_action}")

            # 4. Execution Agent Decision (Optional)
            exec_action = None
            if self.execution_agent:
                obs = self.execution_agent.format_observation(state_data, pipeline_state.l2_supervision)
                exec_action = self.execution_agent.compute_action(obs)
                logger.info(f"Execution Agent Action: {exec_action}")

            # 5. Consolidate Results
            l3_decision = {
                "type": "DRL_DECISION",
                "symbol": symbol,
                "alpha_action": alpha_action.tolist() if hasattr(alpha_action, 'tolist') else alpha_action,
                "risk_action": risk_action.tolist() if hasattr(risk_action, 'tolist') else risk_action,
                "exec_action": exec_action.tolist() if hasattr(exec_action, 'tolist') else exec_action,
                "timestamp": datetime.now().isoformat()
            }
            
            pipeline_state.set_l3_decision(l3_decision)
            logger.info(f"L3 Decision complete. Result: {l3_decision}")
            
        except Exception as e:
            raise PipelineError(f"L3 DRL Decision failed: {e}") from e
            

    async def _run_portfolio_construction(self, pipeline_state: PipelineState):
        """[已实现] 运行投资组合构建。"""
        logger.info("Running PortfolioConstructor...")
        try:
            # [TBD-4 修复] 移除 TBD 注释，PortfolioConstructor 确实从 state 提取信号
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
            
            # [模式 B: 触发]
            # (OrderManager 已经在 PortfolioConstructor 中通过总线接收了目标)
            await self.order_manager.reconcile_portfolio()

            logger.info("OrderManager reconciliation triggered.")
            
        except Exception as e:
            raise PipelineError(f"OrderManager execution failed: {e}") from e

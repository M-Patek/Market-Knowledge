# Phoenix_project/controller/orchestrator.py
# [主人喵的修复 11.11] 实现了 L2->L3 的数据流 TODO。
# [主人喵的修复 11.12] 实现了 TBD-1 (事件过滤) 和 TBD-5 (审计)
# [Code Opt Expert Fix] Task 0.2: Async L3 & Task 1.1: Registry Integration
# [Safety Patch] Implemented Fail-Closed logic for Ledger and Risk Agents.
# [Task 6.1] Macro Regime Injection
# [Task 6.2] Micro-structure Injection (Spread, Imbalance)
# [Task 4.1] Risk Hard Override
# [Task 5.3] Log Sampling for Idle Loops
# [Code Opt Expert Fix] Task 17: Poison Pill Protection (Reliable ACK)
# [Task 005, 006, 010] Real-time Risk Integration, Temporal Safety, Magic String Removal

import logging
import asyncio
from enum import Enum
from datetime import datetime
from omegaconf import DictConfig
from typing import List, Dict, Optional, Any

from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.core.schemas.data_schema import MarketData
from Phoenix_project.context_bus import ContextBus
from Phoenix_project.data_manager import DataManager
from Phoenix_project.cognitive.engine import CognitiveEngine
from Phoenix_project.events.event_distributor import AsyncEventDistributor
from Phoenix_project.events.risk_filter import EventRiskFilter
from Phoenix_project.ai.market_state_predictor import MarketStatePredictor
from Phoenix_project.cognitive.portfolio_constructor import PortfolioConstructor
from Phoenix_project.execution.order_manager import OrderManager
from Phoenix_project.audit_manager import AuditManager
from Phoenix_project.core.exceptions import CognitiveError, PipelineError
# [Task 1.1 Fix] Import get_logger
from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

class RiskAction(str, Enum):
    HALT = "HALT_TRADING"
    CONTINUE = "CONTINUE"

class Orchestrator:
    """
    [已实现]
    系统的主协调器。按顺序运行 L1, L2, L3 认知层，
    管理 PipelineState，并触发最终的投资组合构建和订单执行。
    """

    def __init__(
        self,
        config: DictConfig,
        context_bus: ContextBus,
        data_manager: Optional[DataManager],
        cognitive_engine: CognitiveEngine,
        event_distributor: AsyncEventDistributor,
        event_filter: EventRiskFilter,
        market_state_predictor: MarketStatePredictor,
        portfolio_constructor: PortfolioConstructor,
        order_manager: OrderManager,
        audit_manager: AuditManager,
        trade_lifecycle_manager: Optional[Any] = None,
        # [Fix II.1] 注入 L3 DRL 智能体
        alpha_agent: Optional[Any] = None,
        risk_agent: Optional[Any] = None,  # L3 DRL Agent
        risk_manager: Optional[Any] = None, # Cognitive Risk Manager
        execution_agent: Optional[Any] = None
    ):
        self.config = config
        self.context_bus = context_bus
        self.data_manager = data_manager
        self.cognitive_engine = cognitive_engine
        self.event_distributor = event_distributor
        self.event_filter = event_filter
        self.market_state_predictor = market_state_predictor
        self.portfolio_constructor = portfolio_constructor
        self.order_manager = order_manager
        self.audit_manager = audit_manager
        self.trade_lifecycle_manager = trade_lifecycle_manager
        # [Fix II.1] 存储 L3 智能体
        self.alpha_agent = alpha_agent
        self.risk_agent = risk_agent
        self.risk_manager = risk_manager
        self.execution_agent = execution_agent
        self.risk_initialized = False
        
        # [Task 5.3] Log Sampling Counter
        self.no_event_counter = 0
        
        logger.info("Orchestrator initialized.")

    async def run_main_cycle(self):
        """
        [已实现]
        执行一个完整的认知-决策-行动 (CDA) 循环。
        这是由 Celery beat (LoopManager) 调度的主要入口点。
        """
        start_time = datetime.now()
        # [Optimization] Reduced start log verbosity or moved to debug if needed, keeping info for now
        logger.info(f"Orchestrator main cycle START at {start_time}")
        
        # [Task 17] Scope Initialization: Ensure accessible in finally block
        new_events = []

        pipeline_state = None
        try:
            # 0. 初始化/恢复状态 [Task 2.3 Refactor]
            pipeline_state = await self.context_bus.load_latest_state()
            
            if not pipeline_state:
                logger.info("No previous state found. Creating new PipelineState.")
                pipeline_state = PipelineState()

            # [Phase II Fix] Lazy Initialize RiskManager (Persistence Loading)
            if self.risk_manager and not self.risk_initialized:
                logger.info("Initializing RiskManager (loading persistence & warm-up)...")
                # Default to system symbol if specific list unavailable
                symbols = [self.config.get('trading', {}).get('default_symbol', 'BTC/USD')]
                await self.risk_manager.initialize(symbols)
                self.risk_initialized = True
            
            # [Time Machine] 关键：使用 DataManager 的时间同步 State
            if self.data_manager:
                pipeline_state.update_time(await self.data_manager.get_current_time())
                pipeline_state.step_index += 1

            # [Task 1 Fix] State Sync: Active sync from Ledger (Brain-Body Connection)
            tlm = self.trade_lifecycle_manager or (self.order_manager.trade_lifecycle_manager if hasattr(self.order_manager, 'trade_lifecycle_manager') else None)
            
            # [Fail-Closed Patch] The Zombie Portfolio Fix
            if tlm:
                real_portfolio = tlm.get_current_portfolio_state({}, timestamp=pipeline_state.current_time) 
                pipeline_state.update_portfolio_state(real_portfolio)
                logger.info(f"Synced portfolio state from Ledger: {len(real_portfolio.positions)} positions.")
            else:
                # CRITICAL: Do not run without a ledger.
                raise RuntimeError("TradeLifecycleManager (Ledger) not available. Critical system dependency missing.")

            # 1. 从事件分发器（Redis）获取新事件
            # [Phase 0 Fix] Use native async call
            new_events = await self.event_distributor.get_pending_events()
            if not new_events:
                # [Task 5.3] Log Sampling: Only log "No events" once every 100 ticks
                self.no_event_counter += 1
                if self.no_event_counter % 100 == 0:
                    logger.info("No new events retrieved. Cycle complete. (Log sampled 1/100)")
                return
            
            # Reset counter on activity
            self.no_event_counter = 0
            logger.info(f"Retrieved {len(new_events)} new events from EventDistributor.")
            pipeline_state.set_raw_events(new_events)

            # [Task 005] Ghost Risk Manager Fix: Update Risk State immediately
            if self.risk_manager:
                for event in new_events:
                    # Check if event looks like market data
                    if event.get("symbol") and "close" in event:
                        try:
                            md = MarketData(**event)
                            await self.risk_manager.on_market_data(md)
                        except Exception as e:
                            logger.warning(f"RiskManager update failed for event: {e}")

            # [TBD-1 修复] 过滤事件
            logger.info(f"Filtering {len(new_events)} events...")
            filtered_events = self.event_filter.filter_batch(new_events)
            
            if not filtered_events:
                logger.info("No events remaining after filtering. Cycle complete.")
                # [Task 17] Redundant Ack Removed: Handled in finally block
                return
            
            logger.info(f"{len(filtered_events)} events remaining after filtering.")

            # 2. 运行 L1 认知层 (并行)
            await self._run_l1_cognition(pipeline_state, filtered_events)

            # 3. 运行 L2 监督层 (并行)
            await self._run_l2_supervision(pipeline_state)
            
            # 3.5. 运行 L2 融合/事实检查 (Task A.1)
            logger.info("Running L2 Cognitive Cycle (Fusion, Fact-Checking, Guardrails)...")
            await self.cognitive_engine.process_cognitive_cycle(pipeline_state)
            
            # 4. 运行市场状态预测
            if self.market_state_predictor:
                await self._run_market_state_prediction(pipeline_state)
            else:
                logger.warning("MarketStatePredictor not available. Skipping.")

            # 5. 运行 L3 决策层 (Reasoning Ensemble -> DRL)
            await self._run_l3_decision(pipeline_state)
            
            # [Safety] Risk Blockade
            if pipeline_state.l3_decision and pipeline_state.l3_decision.get("risk_action") == RiskAction.HALT.value:
                logger.critical("HALT_TRADING triggered by L3 Risk Agent. Aborting cycle.")
                # [Task 1.2 Fix] Do NOT return immediately. Let PortfolioConstructor handle Emergency Liquidation.
                # proceed to _run_portfolio_construction which now handles HALT logic.
                pass
            
            if not pipeline_state.l3_decision:
                logger.warning("L3 DRL Agents did not produce a decision. Cycle ending.")
                if pipeline_state:
                    await self.audit_manager.audit_event(
                        event_type="DECISION_FAILURE",
                        details={"reason": "L3 DRL Agents did not produce a decision."},
                        pipeline_state=pipeline_state
                    )
                return

            # 6. 运行认知->执行 转换 (投资组合构建)
            await self._run_portfolio_construction(pipeline_state)

            # 7. 运行执行 (订单管理器)
            await self._run_execution(pipeline_state)
            
            # [Task 17] Redundant Ack Removed: Handled in finally block

        except CognitiveError as e:
            logger.error(f"CognitiveEngine failed with a known error: {e}", exc_info=True)
            if pipeline_state:
               await self.audit_manager.audit_error(e, "CognitiveEngine", pipeline_state)
            
        except PipelineError as e:
            logger.error(f"Orchestrator pipeline failed: {e}", exc_info=True)
            if pipeline_state:
               await self.audit_manager.audit_error(e, "OrchestratorPipeline", pipeline_state)
            
        except Exception as e:
            logger.critical(f"Orchestrator main cycle failed: {e}", exc_info=True)
            if pipeline_state:
               await self.audit_manager.audit_error(e, "OrchestratorFatal", pipeline_state)
            
            # [Task 17] Emergency Shutdown & Re-raise (Prevent Exception Swallowing)
            await self.portfolio_constructor.emergency_shutdown()
            raise

        finally:
            # [Task 17] Poison Pill Protection: Ensure events are ACKed even on failure
            if new_events:
                try:
                    await self.event_distributor.ack_events(new_events)
                    logger.info("Events acknowledged in finally block.")
                except Exception as e:
                    logger.error(f"Failed to ack events: {e}")

            # 8. 审计
            if pipeline_state:
                self.audit_manager.log_cycle(pipeline_state)
                logger.info("Pipeline state logged to AuditManager.")
                await self.context_bus.save_state(pipeline_state)
                
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(f"Orchestrator main cycle END. Duration: {duration:.2f}s")


    async def _run_l1_cognition(self, pipeline_state: PipelineState, filtered_events: List[Dict]):
        """[已实现] 运行 L1 认知智能体。"""
        logger.info("Running L1 Cognition Layer...")
        try:
            # [Fix Phase I] Pass pipeline_state to run_l1_cognition
            l1_insights = await self.cognitive_engine.run_l1_cognition(
                filtered_events,
                pipeline_state
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
                pipeline_state.raw_events
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
        """[Task 0.2] 运行 L3 决策 (异步 DRL 智能体)。"""
        if not pipeline_state.l1_insights or not pipeline_state.market_state:
            logger.warning("Skipping L3 Decision: Missing L1 insights or Market state.")
            return
        
        logger.info("Running L3 Decision Layer (DRL Agents)...")
        
        try:
            # 1. Prepare Context / State Data
            task_query = pipeline_state.get_main_task_query()
            symbol = task_query.get("symbol") or self.config.get('trading.default_symbol', "BTC/USD")
            
            # [Task 006] Temporal Leakage Fix: Use current batch data only
            market_data = None
            for event in pipeline_state.raw_events:
                if event.get("symbol") == symbol and "close" in event:
                    try:
                        market_data = MarketData(**event)
                        break # Use the first matching event
                    except:
                        continue
            
            # [Task 4.2 Fix] News Blindness Fallback: Fetch from cache if batch lacks price
            if not market_data and self.data_manager:
                market_data = await self.data_manager.get_latest_market_data(symbol)
                if market_data:
                    logger.info(f"Using cached market data for {symbol} to enable Event-Driven decision.")

            if not market_data:
                logger.warning(f"L3 Decision Skipped: No market data for {symbol} in batch or cache.")
                return
            
            price = float(market_data.close) if market_data else 0.0
            
            # [Task 6.2] Micro-structure approximations
            spread = 0.0
            if market_data and market_data.close > 0:
                spread = float(market_data.high - market_data.low) / float(market_data.close)
            
            depth_imbalance = 0.0 
            
            pf_state = pipeline_state.portfolio_state
            balance = float(pf_state.cash) if pf_state else 0.0
            holdings = 0.0
            if pf_state and symbol in pf_state.positions:
                holdings = float(pf_state.positions[symbol].quantity)

            state_data = {
                "balance": balance,
                "holdings": holdings,
                "price": price,
                "symbol": symbol,
                "spread": spread,           # [Task 6.2]
                "depth_imbalance": depth_imbalance # [Task 6.2]
            }

            # 2. Alpha Agent Decision (Async)
            alpha_action = None
            if self.alpha_agent:
                obs = self.alpha_agent.format_observation(state_data, pipeline_state.latest_fusion_result, pipeline_state.market_state) 
                alpha_action = await self.alpha_agent.compute_action(obs)
                logger.info(f"Alpha Agent Action: {alpha_action}")

            # 3. Risk Agent Decision (Async)
            # [Fail-Closed Patch] The Ghost Risk Agent Fix
            risk_action = RiskAction.HALT.value
            if self.risk_agent:
                obs = self.risk_agent.format_observation(state_data, pipeline_state.latest_fusion_result, pipeline_state.market_state)
                # [Task 4.1] Pass fusion_result for Hard Override
                raw_risk_action = await self.risk_agent.compute_action(obs, fusion_result=pipeline_state.latest_fusion_result)
                
                raw_val = raw_risk_action[0] if (hasattr(raw_risk_action, '__len__') and len(raw_risk_action) > 0) else raw_risk_action
                # [Phase II Fix] 1.0 = HALT
                risk_action = RiskAction.HALT.value if float(raw_val) > 0.5 else RiskAction.CONTINUE.value
                logger.info(f"Risk Agent Action (Translated): {risk_action}")

            # 4. Execution Agent Decision (Async)
            exec_action = None
            if self.execution_agent:
                obs = self.execution_agent.format_observation(state_data, pipeline_state.latest_fusion_result, pipeline_state.market_state)
                exec_action = await self.execution_agent.compute_action(obs)
                logger.info(f"Execution Agent Action: {exec_action}")

            # 5. Consolidate Results
            l3_decision = {
                "type": "DRL_DECISION",
                "symbol": symbol,
                "alpha_action": alpha_action.tolist() if hasattr(alpha_action, 'tolist') else alpha_action,
                "risk_action": risk_action,
                "exec_action": exec_action.tolist() if hasattr(exec_action, 'tolist') else exec_action,
                "timestamp": datetime.now().isoformat()
            }
            
            pipeline_state.set_l3_decision(l3_decision)
            
            if alpha_action is not None:
                try:
                    # [Task 4.1 Fix] Polymorphic Signal Handling (Dict vs Scalar)
                    if isinstance(alpha_action, dict):
                        clean_signals = {k: float(v) for k, v in alpha_action.items()}
                        pipeline_state.set_l3_alpha_signal(clean_signals)
                    else:
                        val = float(alpha_action) if not hasattr(alpha_action, '__len__') else float(alpha_action[0])
                        pipeline_state.set_l3_alpha_signal({symbol: val})
                except (ValueError, TypeError, IndexError) as e:
                    logger.warning(f"Could not convert alpha_action {alpha_action} to signal: {e}")

            if market_data:
                pipeline_state.set_market_data_batch([market_data])
            
            logger.info(f"L3 Decision complete. Result: {l3_decision}")
            
        except Exception as e:
            raise PipelineError(f"L3 DRL Decision failed: {e}") from e
            

    async def _run_portfolio_construction(self, pipeline_state: PipelineState):
        """[已实现] 运行投资组合构建。"""
        logger.info("Running PortfolioConstructor...")
        try:
            # [Fix] Await async construct_portfolio
            target_portfolio = await self.portfolio_constructor.construct_portfolio(
                pipeline_state
            )
            if target_portfolio:
                pipeline_state.set_target_portfolio(target_portfolio)
                logger.info("PortfolioConstructor complete. Target portfolio generated.")
            else:
                 logger.warning("PortfolioConstructor returned None (Risk Halt). Execution skipped.")
        except Exception as e:
            raise PipelineError(f"PortfolioConstructor failed: {e}") from e


    async def _run_execution(self, pipeline_state: PipelineState):
        """[已实现] 运行订单管理器执行。"""
        if not pipeline_state.target_portfolio:
            logger.info("Skipping Execution: No target portfolio available.")
            return
            
        logger.info("Running OrderManager (Execution)...")
        try:
            await self.order_manager.reconcile_portfolio(pipeline_state)
            logger.info("OrderManager reconciliation triggered.")
        except Exception as e:
            raise PipelineError(f"OrderManager execution failed: {e}") from e

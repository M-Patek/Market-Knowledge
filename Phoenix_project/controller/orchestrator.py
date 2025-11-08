"""
Orchestrator
协调 Phoenix 系统的主要数据和逻辑流。
"""
import asyncio # <-- [新] 添加 asyncio
from typing import List, Dict, Any
from datetime import datetime # [阶段 4 变更] 导入 datetime

from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.core.exceptions import CognitiveError # <-- [新] 导入异常
from Phoenix_project.data_manager import DataManager
from Phoenix_project.events.risk_filter import EventRiskFilter
# [主人喵的修复 2] 导入 EventDistributor
from Phoenix_project.events.event_distributor import EventDistributor
# [主人喵的修复 2] 移除 StreamProcessor
from Phoenix_project.cognitive.engine import CognitiveEngine
from Phoenix_project.cognitive.portfolio_constructor import PortfolioConstructor
from Phoenix_project.cognitive.risk_manager import RiskManager
from Phoenix_project.execution.order_manager import OrderManager
from Phoenix_project.execution.trade_lifecycle_manager import TradeLifecycleManager
from Phoenix_project.snapshot_manager import SnapshotManager
from Phoenix_project.metrics_collector import MetricsCollector
from Phoenix_project.audit_manager import AuditManager
from Phoenix_project.controller.error_handler import ErrorHandler
from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

class Orchestrator:
    """
    管理整个 AI 交易系统的端到端生命周期，
    从数据摄取到执行。
    """
    
    def __init__(
        self,
        pipeline_state: PipelineState,
        data_manager: DataManager,
        event_filter: EventRiskFilter,
        # [主人喵的修复 2] 移除 stream_processor，添加 event_distributor
        event_distributor: EventDistributor,
        cognitive_engine: CognitiveEngine,
        portfolio_constructor: PortfolioConstructor,
        risk_manager: RiskManager,
        order_manager: OrderManager,
        trade_lifecycle_manager: TradeLifecycleManager,
        snapshot_manager: SnapshotManager,
        metrics_collector: MetricsCollector,
        audit_manager: AuditManager,
        error_handler: ErrorHandler
    ):
        self.pipeline_state = pipeline_state
        self.data_manager = data_manager
        self.event_filter = event_filter
        # [主人喵的修复 2] 存储 event_distributor
        self.event_distributor = event_distributor
        self.cognitive_engine = cognitive_engine
        self.portfolio_constructor = portfolio_constructor
        self.risk_manager = risk_manager
        self.order_manager = order_manager
        self.trade_lifecycle_manager = trade_lifecycle_manager
        self.snapshot_manager = snapshot_manager
        self.metrics_collector = metrics_collector
        self.audit_manager = audit_manager
        self.error_handler = error_handler
        
        self.is_running = False
        logger.info("Orchestrator initialized.")

    def _check_for_events(self) -> List[Dict[str, Any]]:
        """
        [主人喵的修复 2]
        从 EventDistributor (Redis 队列) 批量拉取待处理事件。
        """
        try:
            # (默认拉取最多 100 个事件，以避免阻塞循环)
            pending_events = self.event_distributor.get_pending_events(max_events=100)
            
            if pending_events:
                logger.info(f"Retrieved {len(pending_events)} new events from EventDistributor.")
                
            # (TODO: 在这里添加事件的初步验证或 Schema 检查)
            
            return pending_events
            
        except Exception as e:
            logger.error(f"Failed to check for events: {e}", exc_info=True)
            return []

    def run_main_cycle(self):
        """
        执行一个单独的、离散的系统运行周期。
        (这是由 Celery worker 调用的)
        
        [阶段 4 重构]
        - 移除对 execute_orders 和 update_portfolio 的调用。
        - 添加对 order_manager.process_target_portfolio 的调用。
        """
        if self.is_running:
            logger.warning("Main cycle already in progress. Skipping.")
            return

        self.is_running = True
        logger.info("--- Orchestrator Main Cycle START ---")
        
        try:
            # 1. 事件摄取 (Event Ingestion)
            # [主人喵的修复 2] 从 EventDistributor (Redis) 拉取，而不是 Kafka
            new_events = self._check_for_events()
            
            # (如果需要，也可以安排数据管理器的主动拉取)
            # self.data_manager.pull_scheduled_data()

            # 2. 事件过滤 (Event Filtering)
            # (FIX E1)
            filtered_events = self.event_filter.apply_all_filters(new_events)
            if not filtered_events and not new_events: # [修复] 如果有事件但都被过滤了，也许我们仍想运行
                 logger.info("No new events found. Cycle END.")
                 self.is_running = False
                 return
            elif not filtered_events:
                logger.info("No significant events after filtering, but processing cycle anyway...")
            else:
                 logger.info(f"Processing {len(filtered_events)} filtered events.")


            # 3. 状态更新 (State Update)
            self.pipeline_state.start_new_cycle(filtered_events)
            
            # (FIX E2)
            # 4. 认知引擎 (Cognitive Engine)
            # [修复] 使用 asyncio.run() 来调用异步的 cognitive_engine
            fusion_result = None
            try:
                logger.info("Calling async CognitiveEngine...")
                
                # 关键修复：从同步的 Celery 任务中，启动一个新的事件循环
                # 来运行并等待异步的 process_cognitive_cycle 方法
                cognitive_result = asyncio.run(
                    self.cognitive_engine.process_cognitive_cycle(self.pipeline_state)
                )
                
                # cognitive_engine 返回一个字典，"final_decision" 键包含 FusionResult
                fusion_result = cognitive_result.get("final_decision")
                
                # (可选) 存储事实检查报告
                fact_check_report = cognitive_result.get("fact_check_report")
                if fact_check_report:
                    self.pipeline_state.update_value("last_fact_check_report", fact_check_report)

            except CognitiveError as e:
                # 捕获来自 cognitive_engine 的已知业务逻辑错误
                logger.error(f"CognitiveEngine failed with a known error: {e}")
                self.is_running = False
                self.audit_manager.log_cycle(self.pipeline_state) # 记录失败的周期
                logger.info("--- Orchestrator Main Cycle END (Cognitive Error) ---")
                return
            except Exception as e:
                # 捕获 asyncio.run() 或其他未知的基础设施错误
                logger.critical(f"Failed to run CognitiveEngine: {e}", exc_info=True)
                self.error_handler.handle_critical_error(e, self.pipeline_state)
                self.is_running = False
                return # 退出循环
            
            # 保持原有的检查，以防 cognitive_result 成功返回，但 "final_decision" 为 None
            if not fusion_result:
                # (E2.b) 认知引擎没有产生决策
                logger.warning("Cognitive Engine ran successfully but did not produce a final decision.")
                self.is_running = False
                # (E2.c) 我们仍然需要记录审计
                self.audit_manager.log_cycle(self.pipeline_state)
                logger.info("--- Orchestrator Main Cycle END (No Decision) ---")
                return

            # 5. 投资组合构建 (Portfolio Construction)
            # [主人喵的修复] 获取一次当前状态，供后续步骤使用
            # [阶段 4 变更] TLM 现在通过回调更新，所以我们需要最新的市场数据
            # ... (在步骤 6 中获取)
            
            # [阶段 4 变更] 我们需要在这里获取价格，以便 TLM 可以计算当前状态
            # all_symbols_for_pricing = set() # TODO: 从 fusion_result 和 TLM 获取
            # (这个逻辑流有点问题，TLM 需要价格来 get_state，
            # 但我们可能还没有价格。假设 TLM 可以处理)
            
            # [阶段 4 修复] 我们必须先获取价格，才能构造
            # current_portfolio_state = self.trade_lifecycle_manager.get_current_portfolio_state({}) # 传入空价格
            
            # --- [主人喵的修复] 解决数据依赖缺陷：在执行前获取所有最新价格 ---
            
            # 收集所有相关符号
            current_portfolio_for_symbols = self.trade_lifecycle_manager.get_current_portfolio_state({})
            current_symbols = set(current_portfolio_for_symbols.positions.keys())
            target_symbols_from_fusion = {t.symbol for t in fusion_result.targets}
            all_symbols = current_symbols.union(target_symbols_from_fusion)

            market_prices: Dict[str, float] = {}
            for symbol in all_symbols:
                # 假设 data_manager 能获取到最新的 L1 数据
                market_data = self.data_manager.get_latest_market_data(symbol)
                if market_data and market_data.close > 0:
                    market_prices[symbol] = market_data.close
                else:
                    logger.warning(f"Orchestrator: Could not retrieve latest market price for {symbol}.")
            # --- 结束修复 ---

            # [阶段 4 变更] 在构造前，使用新价格更新 TLM 的市值
            self.trade_lifecycle_manager.mark_to_market(datetime.utcnow(), market_prices)
            # 并获取最终的 "当前" 状态
            current_portfolio_state = self.trade_lifecycle_manager.get_current_portfolio_state(market_prices)

            
            target_portfolio = self.portfolio_constructor.construct(
                fusion_result,
                current_portfolio_state
            )

            # 6. 风险管理 (Risk Management)
            final_portfolio, risk_report = self.risk_manager.evaluate_and_adjust(
                target_portfolio,
                self.pipeline_state
            )
            
            # [阶段 4 修复] 重新获取价格，因为 risk_manager 可能添加了对冲（例如 SPY）
            final_target_symbols = {p.symbol for p in final_portfolio.positions}
            all_symbols_final = all_symbols.union(final_target_symbols)
            
            # [修复] 仅获取缺失的价格
            missing_symbols = all_symbols_final - set(market_prices.keys())
            for symbol in missing_symbols:
                 market_data = self.data_manager.get_latest_market_data(symbol)
                 if market_data and market_data.close > 0:
                    market_prices[symbol] = market_data.close
                 else:
                    logger.warning(f"Orchestrator: Could not retrieve latest market price for risk-added symbol {symbol}.")

            # [阶段 4 变更] 再次更新 TLM 市值
            self.trade_lifecycle_manager.mark_to_market(datetime.utcnow(), market_prices)
            current_portfolio_state_final = self.trade_lifecycle_manager.get_current_portfolio_state(market_prices)


            # 7. 执行 (Execution)
            # [阶段 4 重构]
            # (FIX E3) 移除 generate_orders
            # orders = self.order_manager.generate_orders(...)
            
            # (FIX E4) 移除 execute_orders
            # executed_trades = self.order_manager.execute_orders(orders, market_prices)

            # [阶段 4 新增] 调用新的统一方法
            self.order_manager.process_target_portfolio(
                current_portfolio_state_final,
                final_portfolio,
                market_prices
            )

            # 8. 状态更新 (TLM)
            # [阶段 4 重构] 移除
            # TLM 现在通过 OrderManager 的 _on_fill 回调异步更新
            # self.trade_lifecycle_manager.update_portfolio(executed_trades)
            
            # (FIX E9)
            # 9. 监控 & 审计
            self.metrics_collector.collect_all(
                self.pipeline_state,
                fusion_result,
                risk_report,
                self.trade_lifecycle_manager.get_current_portfolio_state(market_prices) # 获取最新状态
            )
            self.audit_manager.log_cycle(self.pipeline_state)
            
            # 10. 快照 (Snapshot)
            self.snapshot_manager.save_state(
                self.pipeline_state,
                self.trade_lifecycle_manager.get_current_portfolio_state(market_prices)
            )

        except Exception as e:
            # (FIX E10) 捕获主循环中的所有异常
            logger.critical(f"Orchestrator main cycle failed: {e}", exc_info=True)
            self.error_handler.handle_critical_error(e, self.pipeline_state)
        
        finally:
            self.is_running = False
            logger.info("--- Orchestrator Main Cycle END ---")

    def shutdown(self):
        """
        (可选) 安全关闭 Orchestrator。
        """
        logger.info("Orchestrator shutting down...")
        self.is_running = False
        # (清理其他资源)

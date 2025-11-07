"""
Orchestrator
协调 Phoenix 系统的主要数据和逻辑流。
"""
from typing import List, Dict, Any

from Phoenix_project.core.pipeline_state import PipelineState
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
            if not filtered_events:
                logger.info("No significant events after filtering. Cycle END.")
                self.is_running = False
                return

            # 3. 状态更新 (State Update)
            self.pipeline_state.start_new_cycle(filtered_events)
            
            # (FIX E2)
            # 4. 认知引擎 (Cognitive Engine)
            # (假设 cognitive_engine.run() 是一个同步/阻塞的调用)
            fusion_result = self.cognitive_engine.run(self.pipeline_state)
            
            if not fusion_result:
                # (E2.b) 认知引擎没有产生决策
                logger.warning("Cognitive Engine did not produce a final decision.")
                self.is_running = False
                # (E2.c) 我们仍然需要记录审计
                self.audit_manager.log_cycle(self.pipeline_state)
                logger.info("--- Orchestrator Main Cycle END (No Decision) ---")
                return

            # 5. 投资组合构建 (Portfolio Construction)
            target_portfolio = self.portfolio_constructor.construct(
                fusion_result,
                self.trade_lifecycle_manager.get_current_portfolio_state()
            )

            # 6. 风险管理 (Risk Management)
            final_portfolio, risk_report = self.risk_manager.evaluate_and_adjust(
                target_portfolio,
                self.pipeline_state
            )
            
            # 7. 执行 (Execution)
            # (FIX E3)
            orders = self.order_manager.generate_orders(
                self.trade_lifecycle_manager.get_current_portfolio_state(),
                final_portfolio
            )
            
            # (FIX E4)
            executed_trades = self.order_manager.execute_orders(orders)

            # 8. 状态更新 (TLM)
            self.trade_lifecycle_manager.update_portfolio(executed_trades)
            
            # (FIX E9)
            # 9. 监控 & 审计
            self.metrics_collector.collect_all(
                self.pipeline_state,
                fusion_result,
                risk_report,
                self.trade_lifecycle_manager.get_current_portfolio_state()
            )
            self.audit_manager.log_cycle(self.pipeline_state)
            
            # 10. 快照 (Snapshot)
            self.snapshot_manager.save_state(
                self.pipeline_state,
                self.trade_lifecycle_manager.get_current_portfolio_state()
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

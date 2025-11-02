import time
import asyncio
import datetime  # <--- 添加了缺失的导入
from typing import Dict, Any, Optional
from core.pipeline_state import PipelineState
from config.loader import ConfigLoader
from monitor.logging import get_logger
from controller.scheduler import Scheduler
from controller.error_handler import ErrorHandler
from data_manager import DataManager
from events.event_distributor import EventDistributor
from cognitive.engine import CognitiveEngine
from cognitive.portfolio_constructor import PortfolioConstructor
from cognitive.risk_manager import RiskManager
from execution.order_manager import OrderManager
from audit_manager import AuditManager
from metrics_collector import MetricsCollector
from snapshot_manager import SnapshotManager

logger = get_logger(__name__)

class Orchestrator:
    """
    The central coordinator of the entire system.
    It owns all major components and triggers the main processing
    cycle in response to events (e.g., from the Scheduler).
    """

    def __init__(
        self,
        config_loader: ConfigLoader,
        pipeline_state: PipelineState,
        data_manager: DataManager,
        event_distributor: EventDistributor,
        cognitive_engine: CognitiveEngine,
        portfolio_constructor: PortfolioConstructor,
        risk_manager: RiskManager,
        order_manager: OrderManager,
        audit_manager: AuditManager,
        metrics_collector: MetricsCollector,
        snapshot_manager: SnapshotManager,
        error_handler: ErrorHandler,
    ):
        self.config_loader = config_loader
        self.pipeline_state = pipeline_state
        self.data_manager = data_manager
        self.event_distributor = event_distributor
        self.cognitive_engine = cognitive_engine
        self.portfolio_constructor = portfolio_constructor
        self.risk_manager = risk_manager
        self.order_manager = order_manager
        self.audit_manager = audit_manager
        self.metrics_collector = metrics_collector
        self.snapshot_manager = snapshot_manager
        self.error_handler = error_handler

        self.scheduler = Scheduler(self, config_loader.get_config("scheduler"))
        self._is_running_cycle = False
        logger.info("Orchestrator initialized with all components.")

    def start_scheduler(self):
        """Starts the job scheduler (for live mode)."""
        logger.info("Starting scheduler...")
        self.scheduler.start()

    def stop_scheduler(self):
        """Stops the job scheduler."""
        logger.info("Stopping scheduler...")
        self.scheduler.stop()

    async def run_main_cycle(self, decision_id: Optional[str] = None):
        """
        Executes one full processing cycle, from cognition to execution.
        This is the "heartbeat" of the system, triggered by the scheduler.
        """
        if self._is_running_cycle:
            logger.warning("Main cycle triggered but a previous cycle is still running. Skipping.")
            return
            
        self._is_running_cycle = True
        start_time = time.perf_counter()
        
        if not decision_id:
            decision_id = self.audit_manager.generate_decision_id()
        
        await self.pipeline_state.update_state({
            "current_decision_id": decision_id,
            "cycle_start_time": datetime.datetime.utcnow()  # <--- 修正了调用
        })
        
        logger.info(f"--- Starting Main Cycle: {decision_id} ---")

        try:
            # 0. Portfolio/Risk Health Check
            self.risk_manager.check_portfolio_risk(self.pipeline_state)
            if self.risk_manager.circuit_breaker_tripped:
                logger.critical(f"Cycle aborted: Circuit breaker is tripped: {self.risk_manager.circuit_breaker_reason}")
                await self.audit_manager.audit_event(
                    "CYCLE_ABORTED", {"reason": "Circuit breaker tripped"}, self.pipeline_state
                )
                self._is_running_cycle = False
                return # Do not proceed

            # 1. Data Ingestion (triggered by scheduler, assumed done)
            # In a backtest, data would be loaded here.
            # In live, we just assume data has arrived.
            
            # 2. Cognitive Cycle
            logger.debug(f"[{decision_id}] Running CognitiveEngine...")
            cognitive_output = await self.cognitive_engine.process_cognitive_cycle(self.pipeline_state)
            
            if "error" in cognitive_output:
                raise Exception(f"CognitiveEngine failed: {cognitive_output['error']}")

            fusion_result = cognitive_output["final_decision"]
            fusion_result.decision_id = decision_id # Stamp the ID
            
            # 3. Audit the Cognitive Decision (crucial)
            await self.audit_manager.audit_decision_cycle(
                self.pipeline_state, fusion_result, decision_id
            )

            # 4. Portfolio Construction (Signal & Order Generation)
            logger.debug(f"[{decision_id}] Running PortfolioConstructor...")
            signal = self.portfolio_constructor.generate_signal(fusion_result)
            
            # 5. Pre-Signal Risk Check
            rejection_reason = await self.risk_manager.validate_signal(signal, self.pipeline_state)
            if rejection_reason:
                logger.warning(f"Signal for {signal.symbol} rejected by RiskManager: {rejection_reason}")
                await self.audit_manager.audit_event(
                    "SIGNAL_REJECTED", {"signal": signal.model_dump(), "reason": rejection_reason}, self.pipeline_state
                )
            else:
                # 6. Generate Orders
                orders = self.portfolio_constructor.generate_orders_from_signal(
                    signal, self.pipeline_state
                )
                
                # 7. Pre-Trade Risk Check & Execution
                for order in orders:
                    order_rejection = await self.risk_manager.validate_order(order, self.pipeline_state)
                    if order_rejection:
                        logger.warning(f"Order for {order.symbol} rejected by RiskManager: {order_rejection}")
                        await self.audit_manager.audit_event(
                            "ORDER_REJECTED", {"order": order.model_dump(), "reason": order_rejection}, self.pipeline_state
                        )
                    else:
                        # 8. Submit Order for Execution
                        logger.info(f"Submitting valid order for {order.symbol} ({order.quantity}).")
                        await self.order_manager.submit_order(order)
                        
            # 9. End-of-cycle tasks
            end_time = time.perf_counter()
            cycle_time_ms = (end_time - start_time) * 1000
            
            await self.pipeline_state.update_state({
                "last_cycle_time_ms": cycle_time_ms,
                "last_successful_cycle_time": datetime.datetime.utcnow() # <--- 修正了调用
            })
            
            self.metrics_collector.gauge("cycle.time_ms", cycle_time_ms)
            self.metrics_collector.increment("cycle.success")
            
            await self.snapshot_manager.take_snapshot(self.pipeline_state, "cycle_end")
            
            logger.info(f"--- Main Cycle {decision_id} Complete ({cycle_time_ms:.2f} ms) ---")
            
            # Reset error count for the main cycle component
            self.error_handler.reset_failure_count("run_main_cycle")

        except Exception as e:
            end_time = time.perf_counter()
            cycle_time_ms = (end_time - start_time) * 1000
            
            logger.error(f"--- Main Cycle {decision_id} FAILED ({cycle_time_ms:.2f} ms): {e} ---", exc_info=True)
            self.metrics_collector.increment("cycle.failure")
            
            # Log to central error handler
            await self.error_handler.handle_error(e, "run_main_cycle", {"decision_id": decision_id})
            
            # Also log to audit trail
            await self.audit_manager.audit_error(e, "run_main_cycle", self.pipeline_state, decision_id)

        finally:
            self._is_running_cycle = False


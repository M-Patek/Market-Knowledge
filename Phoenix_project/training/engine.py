import pandas as pd
# 修复：[FIX-15] 添加 'Callable' 用于类型提示
from typing import Dict, Any, Optional, Callable

from ..core.pipeline_state import PipelineState
from ..data.data_iterator import DataIterator
from ..controller.orchestrator import Orchestrator
from ..execution.trade_lifecycle_manager import TradeLifecycleManager
from ..monitor.logging import get_logger
# 修复：[FIX-15] 导入 'render_report' (函数)，而不是 'ReportRenderer' (类)
from ..output.renderer import render_report

logger = get_logger(__name__)

class BacktestingEngine:
    """
    Drives the simulation by iterating through historical data and feeding
    it to the Orchestrator, simulating the passage of time.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        data_iterator: DataIterator,
        orchestrator: Orchestrator,
        trade_lifecycle_manager: TradeLifecycleManager,
        # 修复：[FIX-15] 更新类型提示和参数名称
        report_renderer_func: Callable
    ):
        """
        Initializes the BacktestingEngine.
        
        Args:
            config: The main strategy configuration.
            data_iterator: Provides the historical event stream (data).
            orchestrator: The "brain" that processes events and generates signals.
            trade_lifecycle_manager: Manages the simulated portfolio state.
            report_renderer_func: 生成报告的函数。
        """
        self.config = config
        self.backtest_config = config.get('backtesting', {})
        self.data_iterator = data_iterator
        self.orchestrator = orchestrator
        self.trade_lifecycle_manager = trade_lifecycle_manager
        # 修复：[FIX-15]
        self.report_renderer = report_renderer_func
        
        self.start_date = pd.to_datetime(self.backtest_config.get('start_date'))
        self.end_date = pd.to_datetime(self.backtest_config.get('end_date'))
        
        logger.info(f"BacktestingEngine initialized for period: {self.start_date} to {self.end_date}")

    async def run(self):
        """
        Executes the backtest.
        """
        logger.info("--- Starting Backtest Run ---")
        
        # 1. Load initial historical data for all components
        logger.info(f"Preloading historical data up to {self.start_date}...")
        await self.orchestrator.preload_historical_data(self.start_date)
        
        # 2. Initialize the data iterator for the backtest period
        self.data_iterator.setup(self.start_date, self.end_date)
        
        # 3. Main Event Loop
        logger.info("Starting event loop...")
        event_count = 0
        timestamp = self.start_date # 修复：[FIX-10] 初始化 timestamp
        try:
            # The DataIterator yields (timestamp, [events_at_this_timestamp])
            async for timestamp, events_batch in self.data_iterator:
                
                # A. Update the Orchestrator's market state with the latest data
                # This includes both market data (prices) and news events
                await self.orchestrator.update_state_from_batch(timestamp, events_batch)
                
                # B. Trigger the cognitive pipeline for each *new* market event
                # (Price updates usually don't trigger the full AI pipeline,
                # but MarketEvents (news) do.)
                news_events = [e for e in events_batch if e.get('type') == 'MarketEvent']
                
                for event_data in news_events:
                    # The orchestrator processes the event, runs the AI,
                    # generates a signal, and applies trades.
                    # This is the core "tick" of the strategy.
                    await self.orchestrator.process_event_and_execute(event_data)
                    event_count += 1

                # C. Mark-to-market at the end of the time step
                # (e.g., end of day)
                # We need to get the latest prices from the orchestrator's state
                latest_prices = self.orchestrator.get_latest_prices()
                self.trade_lifecycle_manager.mark_to_market(timestamp, latest_prices)

                if event_count % 100 == 0 and event_count > 0:
                    logger.info(f"Processed {event_count} events. Current sim time: {timestamp}")

        except Exception as e:
            logger.error(f"Backtest loop failed at {timestamp}: {e}", exc_info=True)
            self.orchestrator.audit_manager.log_system_error(e, "backtest_loop")

        finally:
            logger.info(f"--- Backtest Run Finished ---")
            logger.info(f"Total events processed: {event_count}")
            
            # 4. Generate Performance Report
            self.generate_report()

    def generate_report(self):
        """
        Generates and saves the final performance report.
        """
        logger.info("Generating performance report...")
        try:
            pnl_history = self.trade_lifecycle_manager.get_pnl_history()
            trade_log = self.trade_lifecycle_manager.get_trade_log()
            
            if pnl_history.empty:
                logger.warning("No PnL history found. Cannot generate report.")
                return

            # 修复：[FIX-15] 直接调用函数，
            # 而不是 'generate_html_report' 方法
            report_html = self.report_renderer(
                pnl_history=pnl_history,
                trade_log=trade_log,
                config=self.config
            )
            
            report_path = self.backtest_config.get('report_output_path', 'logs/backtest_report.html')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_html)
                
            logger.info(f"Performance report saved to: {report_path}")

        except Exception as e:
            logger.error(f"Failed to generate report: {e}", exc_info=True)

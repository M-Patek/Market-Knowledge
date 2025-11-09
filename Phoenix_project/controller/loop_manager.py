import time
from datetime import datetime
from threading import Thread
from typing import Optional

from phoenix_project.context_bus import ContextBus
from phoenix_project.controller.orchestrator import Orchestrator
from phoenix_project.core.pipeline_state import PipelineState
from phoenix_project.data.data_iterator import DataIterator
from phoenix_project.monitor.logging import get_logger

log = get_logger("LoopManager")


class LoopManager:
    """
    管理主事件循环（实时交易或回测）。
    控制循环的开始、停止、暂停和恢复。
    """

    def __init__(
        self,
        orchestrator: Orchestrator,
        data_iterator: DataIterator,
        pipeline_state: PipelineState,
        context_bus: ContextBus,
        loop_config: dict,
    ):
        self.orchestrator = orchestrator
        self.data_iterator = data_iterator
        self.pipeline_state = pipeline_state
        self.context_bus = context_bus
        self.loop_config = loop_config
        self.loop_mode = loop_config.get("mode", "backtest")  # "live" or "backtest"
        self.loop_thread: Optional[Thread] = None
        self.is_running = False

    def start_loop(self):
        """启动主循环在一个单独的线程中。"""
        if self.is_running:
            log.warning("Loop is already running.")
            return

        log.info(f"Starting event loop in '{self.loop_mode}' mode.")
        self.is_running = True
        self.pipeline_state.resume()

        if self.loop_mode == "live":
            self.loop_thread = Thread(target=self._live_loop, daemon=True)
        else:
            self.loop_thread = Thread(target=self._backtest_loop, daemon=True)

        self.loop_thread.start()

    def stop_loop(self):
        """停止主循环。"""
        if not self.is_running:
            log.warning("Loop is not running.")
            return

        log.info("Stopping event loop...")
        self.is_running = False
        if self.loop_thread:
            self.loop_thread.join(timeout=5)
            if self.loop_thread.is_alive():
                log.error("Loop thread did not terminate gracefully.")
        log.info("Event loop stopped.")

    def pause_loop(self):
        """暂停循环。"""
        log.info("Pausing event loop...")
        self.pipeline_state.pause()

    def resume_loop(self):
        """恢复循环。"""
        log.info("Resuming event loop...")
        self.pipeline_state.resume()

    def _live_loop(self):
        """
        实时交易循环。
        这通常是由外部事件（例如，数据到达）驱动的，或者是一个固定的轮询间隔。
        """
        poll_interval = self.loop_config.get("poll_interval_sec", 60)
        while self.is_running:
            if self.pipeline_state.is_paused():
                log.debug("Loop paused. Waiting...")
                time.sleep(1)
                continue

            try:
                log.debug("Live loop tick...")
                # 
                data_chunk = next(self.data_iterator)

                if data_chunk:
                    # [✅ 优化] 
                    timestamp = data_chunk.get('timestamp', datetime.now())
                    self.pipeline_state.update_timestamp(timestamp)
                    log.info(f"Running pipeline for live data at {timestamp}")
                    
                    self.orchestrator.run_pipeline(data_chunk)
                else:
                    log.debug("No new data in this tick.")

            except StopIteration:
                log.info("Live data iterator finished. Stopping loop.")
                self.is_running = False
            except Exception as e:
                log.error(f"Error in live loop: {e}", exc_info=True)
                self.context_bus.publish_error(e, "Live Loop")

            time.sleep(poll_interval)
        log.info("Live loop terminated.")

    def _backtest_loop(self):
        """
        [✅ 优化] 回测循环。
        移除了 'simplified view' 评论，并增加了时间戳管理和日志记录。
        """
        log.info("Backtest loop starting...")
        tick_count = 0
        try:
            for data_chunk in self.data_iterator:
                if not self.is_running:
                    log.info("Backtest loop stopped by request.")
                    break

                while self.pipeline_state.is_paused():
                    if not self.is_running:
                        break
                    log.debug("Backtest paused. Waiting...")
                    time.sleep(0.5)
                
                if not self.is_running:
                    break

                # [✅ 优化] 提取时间戳用于日志记录和状态更新
                timestamp = data_chunk.get('timestamp')
                if not timestamp:
                    log.warning(f"Data chunk missing timestamp. Assigning current time. Tick: {tick_count}")
                    timestamp = datetime.now()
                
                self.pipeline_state.update_timestamp(timestamp)
                
                log.debug(f"Running pipeline for backtest timestamp: {timestamp}")
                self.orchestrator.run_pipeline(data_chunk)
                tick_count += 1
                
                log.debug(f"Tick {tick_count} complete for {timestamp}. "
                          f"Equity: {self.pipeline_state.portfolio.total_equity:.2f}")

        except StopIteration:
            log.info("Data iterator finished.")
        except Exception as e:
            log.error(f"Error in backtest loop at tick {tick_count}: {e}", exc_info=True)
            self.context_bus.publish_error(e, f"Backtest Loop (Tick {tick_count})")
        
        self.is_running = False
        log.info(f"Backtest loop finished. Total ticks processed: {tick_count}")

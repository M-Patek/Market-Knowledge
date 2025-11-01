import asyncio  # 修复：导入 asyncio
from data_manager import DataManager
from monitor.logging import get_logger
from controller.orchestrator import Orchestrator as PipelineOrchestrator
from registry import registry
from backtesting.engine import BacktestingEngine
from ai.reasoning_ensemble import ReasoningEnsemble

# Configure logger for this module (Layer 12)
logger = get_logger(__name__)

class CognitiveEngine:
    """
    The main engine driving the cognitive simulation.
    It orchestrates data flow and the cognitive pipeline.
    """

    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        # self.l1_orchestrator = L1Orchestrator() # Replaced by Layer 9 orchestrator
        self.pipeline_orchestrator = PipelineOrchestrator()
        self.backtesting_engine: BacktestingEngine = registry.resolve("backtesting_engine") 
        self.reasoning_ensemble: ReasoningEnsemble = registry.resolve("reasoning_ensemble")

        logger.info("CognitiveEngine initialized.")

    def run_simulation(self):
        logger.info("CognitiveEngine: Starting simulation...")
        
        all_signals = [] # Collect signals for backtesting

        for data_event in self.data_manager.stream_data():
            ticker = data_event.get('ticker', 'UNKNOWN') # 修复：从 data_event 中获取 ticker
            logger.info(f"CognitiveEngine: Processing event for {ticker}")
            
            # 修复：使用 asyncio.run() 来正确调用异步的 pipeline
            # 修复：将返回值视为 'pipeline_result' (一个 dict)，而不是 'pipeline_result_state'
            try:
                # 运行异步的 pipeline 并等待其 dict 结果
                pipeline_result = asyncio.run(
                    self.pipeline_orchestrator.run_pipeline(data_event)
                )
            except Exception as e:
                logger.error(f"CognitiveEngine: Pipeline run failed for {ticker}: {e}")
                continue # 继续处理下一个事件
            
            # 修复：从 'pipeline_result' (dict) 中 .get() 数据
            # orchestrator.py 确保了 'ticker' 字段存在于返回的 dict 中
            mock_signal = pipeline_result.get("Signal_generation")
            if not mock_signal:
                # 修复：使用我们之前从 data_event 中获取的 ticker
                mock_signal = {"ticker": ticker, "action": "HOLD", "confidence": 0.5}
            
            all_signals.append(mock_signal)
            # 修复：使用 ticker 变量进行日志记录
            logger.info(f"CognitiveEngine: Pipeline run completed for {ticker}")

        logger.info("CognitiveEngine: Simulation finished.")

        # --- Layer 14: Backtesting Feedback Loop ---
        logger.info("CognitiveEngine: Starting backtesting run...")
        metrics = self.backtesting_engine.run_backtest(all_signals)
        
        logger.info(f"CognitiveEngine: Backtest complete. Metrics: {metrics}")
        logger.info("CognitiveEngine: Feeding metrics back to L2 (ReasoningEnsemble)...")
        
        self.reasoning_ensemble.meta_update(metrics)
        
        logger.info("CognitiveEngine: Layer 14 feedback loop complete.")

    def run_single_event(self, data_event: dict):
        """
        Runs the pipeline for a single event, intended for API calls (Layer 9).
        """
        ticker = data_event.get('ticker', 'UNKNOWN') # 修复：从 data_event 中获取 ticker
        logger.info(f"CognitiveEngine: Processing single event for {ticker}")
        
        # 修复：使用 asyncio.run() 来正确调用异步的 pipeline
        # 修复：将返回值视为 'pipeline_result' (一个 dict)
        try:
            pipeline_result = asyncio.run(
                self.pipeline_orchestrator.run_pipeline(data_event)
            )
            logger.info(f"CognitiveEngine: Single event run completed for {ticker}")
            return pipeline_result # 修复：返回 'pipeline_result' (dict)
        except Exception as e:
            logger.error(f"CognitiveEngine: Single event pipeline run failed for {ticker}: {e}")
            return {"error": str(e), "ticker": ticker}


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
            logger.info(f"CognitiveEngine: Processing event for {data_event.get('ticker')}")
            
            # Run the full pipeline via the orchestrator (Layer 9)
            pipeline_result_state = self.pipeline_orchestrator.run_pipeline(data_event)
            
            # Mock: Assume the pipeline state now contains a signal
            # In a real app, this would come from the 'Signal_generation' stage
            mock_signal = pipeline_result_state.get_data("Signal_generation")
            if not mock_signal:
                mock_signal = {"ticker": data_event.get('ticker'), "action": "HOLD", "confidence": 0.5}
            
            all_signals.append(mock_signal)
            logger.info(f"CognitiveEngine: Pipeline run completed for {pipeline_result_state.ticker}")

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
        logger.info(f"CognitiveEngine: Processing single event for {data_event.get('ticker')}")
        pipeline_result_state = self.pipeline_orchestrator.run_pipeline(data_event)
        logger.info(f"CognitiveEngine: Single event run completed for {pipeline_result_state.ticker}")
        return pipeline_result_state


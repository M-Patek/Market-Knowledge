from typing import Dict, Any, Optional

# 修正：[FIX-ImportError]
# 将所有 `..` 相对导入更改为从项目根目录开始的绝对导入，
# 以匹配项目的标准约定 (如 phoenix_project.py 中所设定的)。
from core.pipeline_state import PipelineState
from core.schemas.fusion_result import FusionResult
from ai.metacognitive_agent import MetacognitiveAgent
# 'portfolio_constructor' 在同一目录 'cognitive/' 下，
# 使用相对导入 `.` 是可以的，但为了统一，我们也使用绝对导入。
from cognitive.portfolio_constructor import PortfolioConstructor, Portfolio
from execution.signal_protocol import StrategySignal
from monitor.logging import get_logger

logger = get_logger(__name__)

class CognitiveEngine:
    """
    The Cognitive Engine is the "brain" of the strategy.
    It encapsulates the MetacognitiveAgent (which runs the AI pipeline)
    and the PortfolioConstructor (which translates AI decisions into targets).
    
    Its main job is to take the current system state and an AI's fusion result,
    and output a concrete StrategySignal (target portfolio weights).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        metacognitive_agent: MetacognitiveAgent,
        portfolio_constructor: PortfolioConstructor
    ):
        """
        Initializes the CognitiveEngine.
        
        Args:
            config: The main strategy configuration.
            metacognitive_agent: The agent responsible for AI reasoning (RAG, ensemble, etc.).
            portfolio_constructor: The module that builds the target portfolio.
        """
        self.config = config
        self.metacognitive_agent = metacognitive_agent
        self.portfolio_constructor = portfolio_constructor
        logger.info("CognitiveEngine initialized.")

    async def run_cognitive_pipeline(
        self, 
        state: PipelineState
    ) -> Optional[FusionResult]:
        """
        Runs the full AI reasoning pipeline on the *triggering event*
        found in the current state.
        
        Args:
            state: The current PipelineState (must contain a 'triggering_event').
            
        Returns:
            Optional[FusionResult]: The output from the MetacognitiveAgent,
                                    or None if no event was processed.
        """
        
        if not state.triggering_event:
            logger.warning("Cognitive pipeline run skipped: No triggering event in state.")
            return None
            
        logger.debug(f"Running cognitive pipeline for event: {state.triggering_event.event_id}")
        
        # The MetacognitiveAgent handles RAG, the agent ensemble,
        # fact-checking, arbitration, and uncertainty calculation.
        fusion_result = await self.metacognitive_agent.process_event(
            event=state.triggering_event,
            market_context=state.get_all_market_data_as_list() # Provide all market data
        )
        
        return fusion_result

    def generate_target_signal(
        self,
        state: PipelineState,
        fusion_result: FusionResult
    ) -> StrategySignal:
        """
        Generates the target portfolio weights based on the AI's decision.
        
        Args:
            state: The current PipelineState.
            fusion_result: The output from the cognitive pipeline run.
            
        Returns:
            StrategySignal: A standardized signal object with target weights.
        """
        
        logger.debug(f"Generating target signal for event: {fusion_result.event_id}")
        
        # The PortfolioConstructor translates the (qualitative) AI decision
        # and (quantitative) uncertainty score into a (quantitative)
        # target portfolio (e.g., {"AAPL": 0.1, "CASH": 0.9}).
        portfolio: Portfolio = self.portfolio_constructor.generate_optimized_portfolio(
            state=state,
            fusion_result=fusion_result
        )
        
        # Package the portfolio into a standard StrategySignal
        signal = StrategySignal(
            strategy_id=state.strategy_name,
            timestamp=state.timestamp,
            target_weights=portfolio.weights,
            metadata={
                "event_id": fusion_result.event_id,
                "ai_decision": fusion_result.final_decision.decision,
                "ai_confidence": fusion_result.final_decision.confidence,
                "cognitive_uncertainty": fusion_result.cognitive_uncertainty,
                **portfolio.metadata # Include metadata from portfolio construction
            }
        )
        
        logger.info(f"Generated signal for {fusion_result.event_id}: "
                    f"Decision: {signal.metadata['ai_decision']}, "
                    f"Uncertainty: {signal.metadata['cognitive_uncertainty']:.2f}")
        
        return signal

}

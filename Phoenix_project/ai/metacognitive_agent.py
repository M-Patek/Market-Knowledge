import asyncio
from typing import Dict, Any, List, Optional
import numpy as np

from ..core.schemas.data_schema import MarketEvent, TickerData
from ..core.schemas.fusion_result import FusionResult, AgentDecision, AgentIO
from .prompt_manager import PromptManager
from .retriever import Retriever
from .source_credibility import SourceCredibilityModel
from ..evaluation.fact_checker import FactChecker
from ..reasoning.compressor import ContextCompressor
from ..api.gemini_pool_manager import GeminiPoolManager
from ..monitor.logging import get_logger

logger = get_logger(__name__)

class MetacognitiveAgent:
    """
    The core AI agent that orchestrates the entire reasoning pipeline.
    It combines RAG, multi-agent ensemble reasoning, fact-checking, and
    metacognitive self-correction to produce a final, uncertainty-aware result.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        prompt_manager: PromptManager,
        retriever: Retriever,
        gemini_pool: GeminiPoolManager,
        source_credibility_model: SourceCredibilityModel,
        fact_checker: FactChecker,
        context_compressor: ContextCompressor
    ):
        self.config = config
        self.prompt_manager = prompt_manager
        self.retriever = retriever
        self.gemini_pool = gemini_pool
        self.source_credibility_model = source_credibility_model
        self.fact_checker = fact_checker
        self.context_compressor = context_compressor
        self.use_metacognition = config.get('ai_ensemble', {}).get('use_metacognition', True)
        self.metacognition_config = config.get('ai_ensemble', {}).get('metacognition', {})
        
        logger.info(f"MetacognitiveAgent initialized. Metacognition enabled: {self.use_metacognition}")

    async def process_event(
        self, 
        event: MarketEvent, 
        market_context: Optional[List[TickerData]] = None
    ) -> FusionResult:
        """
        Main entry point for processing a single market event.
        """
        event_id = event.event_id
        logger.info(f"Processing event: {event_id} - {event.headline}")
        
        pipeline_io = {}
        
        try:
            # 1. Hybrid RAG (Retrieval-Augmented Generation)
            context_bundle, retrieval_metadata = await self.retriever.retrieve_hybrid_context(event, market_context)
            pipeline_io['retrieval'] = retrieval_metadata

            # 2. Context Compression
            compressed_context = await self.context_compressor.compress_context(context_bundle, event.headline)
            pipeline_io['compressed_context'] = compressed_context

            # 3. Source Credibility Assessment
            source_scores = self.source_credibility_model.score_sources(context_bundle)
            pipeline_io['source_scores'] = source_scores

            # 4. Prepare Prompts
            system_prompts = self.prompt_manager.get_all_system_prompts()
            
            # 5. Run initial Agent Ensemble
            # TODO: This logic needs to be encapsulated in EnsembleClient
            # decisions, agent_io = await self.ensemble_client.run_ensemble(compressed_context, system_prompts, event_id)
            # ... temp placeholder ...
            decisions = [] # Placeholder
            agent_io = {}  # Placeholder
            
            pipeline_io['agent_ensemble_io'] = agent_io

            # 6. Fact Checking (if enabled)
            # ... logic for fact-checking ...
            
            # 7. Metacognition / Self-Correction (if enabled)
            if self.use_metacognition and self._should_trigger_metacognition(decisions):
                logger.info(f"Metacognition triggered for event: {event_id}")
                # ... logic for arbitration and re-ranking ...
                pass

            # 8. Synthesize Final Result
            # final_decision, uncertainty = self.synthesize_results(decisions, source_scores)
            final_decision = AgentDecision(agent_id="synthesizer", decision="PLACEHOLDER", confidence=0.0, justification="Not implemented")
            uncertainty = {"cognitive_uncertainty": 1.0}


            logger.info(f"Successfully processed event: {event_id}. Final decision: {final_decision.decision}")
            
            return FusionResult(
                event_id=event_id,
                final_decision=final_decision,
                cognitive_uncertainty=uncertainty.get('cognitive_uncertainty', 1.0),
                agent_decisions=decisions,
                pipeline_io=pipeline_io,
                status="SUCCESS"
            )

        except Exception as e:
            logger.error(f"Error processing event {event_id}: {e}", exc_info=True)
            return FusionResult(
                event_id=event_id,
                final_decision=AgentDecision(
                    agent_id="system_error",
                    decision="ERROR",
                    confidence=0.0,
                    justification=str(e)
                ),
                cognitive_uncertainty=1.0,
                agent_decisions=[],
                pipeline_io=pipeline_io,
                status="ERROR",
                error_message=str(e)
            )

    def _should_trigger_metacognition(self, decisions: List[AgentDecision]) -> bool:
        """
        Determines if the metacognitive (arbitration) step should be triggered
        based on agent disagreement or low confidence.
        """
        if not decisions:
            return False
            
        # Example trigger logic:
        # 1. High disagreement (e.g., mix of BUY, SELL, HOLD)
        unique_decisions = set(d.decision for d in decisions if d.decision not in ["ERROR", "INVALID_RESPONSE"])
        if len(unique_decisions) > self.metacognition_config.get('disagreement_threshold', 1):
            logger.debug("Metacognition trigger: High disagreement.")
            return True
            
        # 2. Low average confidence
        confidences = [d.confidence for d in decisions if d.decision not in ["ERROR", "INVALID_RESPONSE"]]
        if not confidences:
            return False
            
        avg_confidence = np.mean(confidences)
        if avg_confidence < self.metacognition_config.get('confidence_threshold', 0.5):
            logger.debug(f"Metacognition trigger: Low average confidence ({avg_confidence:.2f}).")
            return True
            
        return False

    def synthesize_results(
        self, 
        decisions: List[AgentDecision], 
        source_scores: Dict[str, float]
    ) -> (AgentDecision, Dict[str, float]):
        """
        Fuses all agent decisions and source credibility into a final
        decision and uncertainty score.
        
        This is a placeholder. A real implementation would use a more
        sophisticated Bayesian model.
        """
        
        # Placeholder logic: weighted average
        total_weight = 0
        weighted_score = 0
        valid_justifications = []
        
        decision_map = {"BUY": 1, "HOLD": 0, "SELL": -1}
        
        for d in decisions:
            if d.decision in decision_map:
                weight = d.confidence
                total_weight += weight
                weighted_score += weight * decision_map[d.decision]
                valid_justifications.append(f"[{d.agent_id} @ {d.confidence:.0%}): {d.justification}")
        
        if total_weight == 0:
            final_decision_str = "HOLD"
            final_confidence = 0.0
            cognitive_uncertainty = 1.0
        else:
            final_score = weighted_score / total_weight
            final_confidence = np.mean([d.confidence for d in decisions if d.decision in decision_map])
            
            # Simple discretization
            if final_score > 0.33:
                final_decision_str = "BUY"
            elif final_score < -0.33:
                final_decision_str = "SELL"
            else:
                final_decision_str = "HOLD"
            
            # Calculate cognitive uncertainty
            # Example: 1.0 - (agreement * avg_confidence)
            std_dev = np.std([decision_map[d.decision] for d in decisions if d.decision in decision_map])
            agreement = 1.0 - (std_dev / np.max([1.0, 1.0])) # Normalized std dev
            cognitive_uncertainty = 1.0 - (agreement * final_confidence)


        final_justification = " | ".join(valid_justifications)
        
        final_decision = AgentDecision(
            agent_id="synthesizer",
            decision=final_decision_str,
            confidence=final_confidence,
            justification=final_justification
        )
        
        uncertainty_scores = {
            "cognitive_uncertainty": cognitive_uncertainty,
            "average_confidence": final_confidence,
            "decision_agreement": agreement
        }

        return final_decision, uncertainty_scores

import asyncio
from typing import Any, Dict, List, Optional
import json

from ai.metacognitive_agent import MetacognitiveAgent
from ai.prompt_manager import PromptManager
from api.gateway import APIGateway
from config.loader import ConfigLoader
from core.schemas.fusion_result import FusionResult
from evaluation.arbitrator import Arbitrator
from evaluation.calibrator import Calibrator
from evaluation.critic import Critic
from evaluation.fact_checker import FactChecker
from memory.cot_database import CoTDatabase
from memory.vector_store import VectorStore
from monitor.logging import ESLogger
from reasoning.compressor import ReasoningCompressor
from reasoning.planner import ReasoningPlanner


class FusionSynthesizer:
    """
    Orchestrates the synthesis of information from various sources, incorporating
    planning, metacognition, and evaluation to produce a high-confidence,
    low-uncertainty output.
    """

    def __init__(
        self,
        config_loader: ConfigLoader,
        prompt_manager: PromptManager,
        api_gateway: APIGateway,
        vector_store: VectorStore,
        cot_database: CoTDatabase,
        logger: ESLogger,
    ):
        self.config = config_loader.get_config("synthesizer")
        self.prompt_manager = prompt_manager
        self.api_gateway = api_gateway
        self.vector_store = vector_store
        self.cot_database = cot_database
        self.logger = logger

        # Initialize components
        self.planner = ReasoningPlanner(
            self.config["planner"], self.api_gateway, self.prompt_manager
        )
        self.metacognitive_agent = MetacognitiveAgent(
            self.config["metacognitive_agent"],
            self.api_gateway,
            self.prompt_manager,
        )
        self.compressor = ReasoningCompressor(
            self.config["compressor"], self.api_gateway, self.prompt_manager
        )
        self.critic = Critic(
            self.config["critic"], self.api_gateway, self.prompt_manager
        )
        self.fact_checker = FactChecker(
            self.config["fact_checker"], self.api_gateway, self.prompt_manager
        )
        self.calibrator = Calibrator(self.config["calibrator"])
        self.arbitrator = Arbitrator(
            self.config["arbitrator"], self.api_gateway, self.prompt_manager
        )
        self.logger.log_info("FusionSynthesizer initialized successfully.")

    async def synthesize_event(
        self, event_data: Dict[str, Any]
    ) -> FusionResult:
        """
        Processes a single market event through the full synthesis pipeline.

        Args:
            event_data: The raw event data.

        Returns:
            A FusionResult object containing the synthesized output and metadata.
        """
        self.logger.log_info(
            f"Starting synthesis for event: {event_data.get('event_id', 'N/A')}"
        )
        try:
            # 1. Plan
            plan = await self.planner.generate_plan(event_data)
            self.logger.log_debug(f"Plan generated: {plan}")

            # 2. Execute (Metacognition & Information Gathering)
            (
                metacognitive_trace,
                collected_data,
            ) = await self.metacognitive_agent.execute_search_queries(
                event_data, plan
            )
            self.logger.log_debug(
                f"Metacognitive execution complete. Data collected: {len(collected_data)} items."
            )

            # 3. Compress & Summarize
            compressed_context = await self.compressor.compress_information(
                collected_data
            )
            self.logger.log_debug(
                f"Information compressed. Context length: {len(compressed_context)}"
            )

            # 4. Synthesize Initial Hypothesis (using Metacognitive Agent's synthesis capability)
            initial_hypothesis = (
                await self.metacognitive_agent.synthesize_hypothesis(
                    event_data, compressed_context, metacognitive_trace
                )
            )
            self.logger.log_info(
                f"Initial hypothesis synthesized: {initial_hypothesis['hypothesis']}"
            )

            # 5. Evaluate (Critic, Fact-Check, Calibrate)
            critic_assessment = await self.critic.assess_hypothesis(
                initial_hypothesis
            )
            self.logger.log_debug(f"Critic assessment: {critic_assessment}")

            fact_check_results = await self.fact_checker.verify_facts(
                initial_hypothesis
            )
            self.logger.log_debug(
                f"Fact check results: {fact_check_results}"
            )

            calibrated_confidence = self.calibrator.calibrate_confidence(
                initial_hypothesis["confidence"],
                [critic_assessment, fact_check_results],
            )
            self.logger.log_debug(
                f"Confidence calibrated: {calibrated_confidence}"
            )

            # 6. Arbitrate & Refine
            final_synthesis = await self.arbitrator.arbitrate_and_refine(
                initial_hypothesis,
                [critic_assessment, fact_check_results],
                calibrated_confidence,
            )
            self.logger.log_info(
                f"Final synthesis complete: {final_synthesis['final_assessment']}"
            )

            # 7. Construct Final Result
            result = FusionResult(
                event_id=event_data.get("event_id", "N/A"),
                timestamp=event_data.get("timestamp", ""),
                source=event_data.get("source", ""),
                plan=plan,
                metacognitive_trace=metacognitive_trace,
                compressed_context=compressed_context,
#                initial_hypothesis=initial_hypothesis["hypothesis"],
                initial_hypothesis = json.dumps(initial_hypothesis),
                critic_assessment=critic_assessment,
                fact_check_results=fact_check_results,
                calibrated_confidence=calibrated_confidence,
                final_assessment=final_synthesis["final_assessment"],
                uncertainty_score=final_synthesis["uncertainty_score"],
                key_takeaways=final_synthesis["key_takeaways"],
            )

            # 8. Store CoT and update Vector Store
            await self._store_reasoning(result)

            return result

        except Exception as e:
            self.logger.log_error(
                f"Error during synthesis pipeline for event {event_data.get('event_id', 'N/A')}: {e}",
                exc_info=True,
            )
            # Return a structured error result
            return FusionResult(
                event_id=event_data.get("event_id", "N/A"),
                timestamp=event_data.get("timestamp", ""),
                source=event_data.get("source", ""),
                final_assessment="Synthesis failed due to internal error.",
                uncertainty_score=1.0,
                error_message=str(e),
            )

    async def _store_reasoning(self, result: FusionResult):
        """
        Stores the chain-of-thought (CoT) and updates the vector store.
        """
        try:
            # Store full reasoning trace in CoT database
            await self.cot_database.store_trace(
                result.event_id, result.model_dump()
            )

            # Update vector store with key takeaways
            if result.key_takeaways:
                document = {
                    "id": result.event_id,
                    "content": "\n".join(result.key_takeaways),
                    "metadata": {
                        "source": "fusion_synthesizer",
                        "timestamp": result.timestamp,
                        "confidence": result.calibrated_confidence,
                        "uncertainty": result.uncertainty_score,
                    },
                }
                await self.vector_store.aadd_documents([document])
            self.logger.log_debug(
                f"Reasoning trace and vector store updated for event {result.event_id}"
            )
        except Exception as e:
            self.logger.log_warning(
                f"Failed to store reasoning for event {result.event_id}: {e}",
                exc_info=True,
            )

    async def process_batch(
        self, events: List[Dict[str, Any]]
    ) -> List[FusionResult]:
        """
        Processes a batch of events concurrently.

        Args:
            events: A list of event data dictionaries.

        Returns:
            A list of FusionResult objects.
        """
        self.logger.log_info(
            f"Starting batch synthesis for {len(events)} events."
        )
        tasks = [self.synthesize_event(event) for event in events]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        final_results = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                event_id = events[i].get("event_id", f"unknown_batch_event_{i}")
                self.logger.log_error(
                    f"Error processing event {event_id} in batch: {res}",
                    exc_info=res,
                )
                final_results.append(
                    FusionResult(
                        event_id=event_id,
                        timestamp=events[i].get("timestamp", ""),
                        source=events[i].get("source", ""),
                        final_assessment="Synthesis failed due to batch processing error.",
                        uncertainty_score=1.0,
                        error_message=str(res),
                    )
                )
            else:
                final_results.append(res)

        self.logger.log_info(
            f"Batch synthesis complete. Processed {len(final_results)} results."
        )
        return final_results

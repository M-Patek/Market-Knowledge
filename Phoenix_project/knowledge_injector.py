import asyncio
from typing import Dict, Any, List

from monitor.logging import ESLogger
from core.pipeline_state import PipelineState
from knowledge_graph_service import KnowledgeGraphService
from ai.relation_extractor import RelationExtractor


class KnowledgeInjector:
    """
    Component responsible for processing high-confidence, low-uncertainty
    fusion results and injecting them into the Knowledge Graph.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: ESLogger,
        kg_service: KnowledgeGraphService,
        relation_extractor: RelationExtractor,
    ):
        """
        Initializes the KnowledgeInjector.

        Args:
            config: Configuration dictionary for the injector.
            logger: An instance of ESLogger for logging.
            kg_service: The KnowledgeGraphService for KG interactions.
            relation_extractor: The RelationExtractor for converting text to structured data.
        """
        self.config = config
        self.logger = logger
        self.kg_service = kg_service
        self.relation_extractor = relation_extractor
        self.logger.log_info("KnowledgeInjector initialized.")

    async def process_state(self, state: PipelineState) -> PipelineState:
        """
        Processes the pipeline state. If a valid fusion result exists,
        it extracts relations and injects them into the KG.

        Args:
            state: The current PipelineState object.

        Returns:
            The potentially modified PipelineState object.
        """
        if not state.fusion_result or not state.fusion_result.final_assessment:
            self.logger.log_debug("No valid fusion result in state, skipping injection.")
            return state

        # Check if the uncertainty guard already flagged this
        if state.error:
            self.logger.log_debug(
                f"State has error, skipping injection: {state.error}"
            )
            return state

        self.logger.log_info(
            f"Processing fusion result for event {state.fusion_result.event_id} for KG injection."
        )

        try:
            # 1. Extract structured relations from the fusion result's assessment
            assessment_text = state.fusion_result.final_assessment
            key_takeaways = (
                "\n".join(state.fusion_result.key_takeaways)
                if state.fusion_result.key_takeaways
                else ""
            )
            full_text = f"{assessment_text}\n{key_takeaways}"

            extracted_data = await self.relation_extractor.extract_relations(full_text)

            if not extracted_data:
                self.logger.log_warning(
                    f"No relations extracted for event {state.fusion_result.event_id}. Nothing to inject."
                )
                return state

            # 2. Format data for KG update
            # This logic depends heavily on the KG structure and data models
            kg_update_payload = self._format_for_kg(
                extracted_data, state.fusion_result
            )

            if not kg_update_payload:
                self.logger.log_warning(
                    f"Failed to format extracted data for KG for event {state.fusion_result.event_id}."
                )
                return state

            # 3. Inject into Knowledge Graph
            update_success = await self.kg_service.update_knowledge_graph(
                kg_update_payload
            )

            if update_success:
                self.logger.log_info(
                    f"Successfully injected knowledge for event {state.fusion_result.event_id} into KG."
                )
                state.set_knowledge_injected(True)
            else:
                self.logger.log_error(
                    f"Failed to inject knowledge for event {state.fusion_result.event_id} into KG."
                )
                state.set_knowledge_injected(False)

        except Exception as e:
            self.logger.log_error(
                f"Error during knowledge injection for event {state.fusion_result.event_id}: {e}",
                exc_info=True,
            )
            state.set_knowledge_injected(False)

        return state

    def _format_for_kg(
        self, extracted_data: List[Dict[str, Any]], fusion_result: Any
    ) -> Dict[str, Any]:
        """
        Formats the unstructured data extracted by the RelationExtractor
        into a structured payload for the KnowledgeGraphService.

        This is a placeholder and needs to be implemented based on the actual
        temporal and tabular DB schemas.

        Args:
            extracted_data: Output from RelationExtractor.
            fusion_result: The FusionResult object for context (e.g., timestamps).

        Returns:
            A dictionary formatted for `kg_service.update_knowledge_graph`.
        """
        # Example formatting logic (highly simplified)
        temporal_updates = []
        tabular_inserts = []

        timestamp = fusion_result.timestamp
        source = fusion_result.source

        for item in extracted_data:
            # Simple example:
            if item.get("type") == "market_event" and "entity" in item:
                temporal_updates.append(
                    {
                        "entity": item["entity"],
                        "property": "observed_event",
                        "value": item.get("details", "unknown"),
                        "timestamp": timestamp,
                        "source": source,
                    }
                )
            elif item.get("type") == "price_target" and "ticker" in item:
                tabular_inserts.append(
                    {
                        "table_id": "market_predictions",
                        "row": {
                            "ticker": item["ticker"],
                            "target_price": item.get("value"),
                            "analyst": item.get("source_analyst", source),
                            "timestamp": timestamp,
                        },
                    }
                )

        payload = {}
        if temporal_updates:
            payload["temporal_updates"] = temporal_updates
        if tabular_inserts:
            payload["tabular_inserts"] = tabular_inserts

        return payload

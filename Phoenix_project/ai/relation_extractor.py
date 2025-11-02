"""
Relation Extractor for Knowledge Graph.

Uses an LLM to extract entities (nodes) and relationships (edges)
from unstructured text (e.g., MarketEvent content) to build or
update a Knowledge Graph.
"""
import logging
from typing import List, Dict, Any, Optional

# 修复：使用正确的相对导入
from ..api.gemini_pool_manager import GeminiPoolManager
from ..core.schemas.data_schema import KnowledgeGraph, KGNode, KGRelation, MarketEvent
from .prompt_manager import PromptManager
from .prompt_renderer import PromptRenderer

logger = logging.getLogger(__name__)

class RelationExtractor:
    """
    Extracts entities and relations from text to build a Knowledge Graph.
    """

    def __init__(self,
                 prompt_manager: PromptManager,
                 prompt_renderer: PromptRenderer,
                 gemini_pool: GeminiPoolManager,
                 prompt_name: str = "relation_extractor"): # Example prompt name
        
        self.prompt_manager = prompt_manager
        self.prompt_renderer = prompt_renderer
        self.gemini_pool = gemini_pool
        self.prompt_template = self.prompt_manager.get_prompt(prompt_name)
        
        if not self.prompt_template:
            # Fallback or error if prompt is missing
            logger.warning(f"Prompt '{prompt_name}' not found. Using a basic fallback.")
            self.prompt_template = """
            Extract entities and relations from the text.
            Text: {{ event.content }}
            Respond in JSON format: {"nodes": [...], "relations": [...]}
            """
        
        logger.info("RelationExtractor initialized.")

    async def extract(self, event: MarketEvent) -> Optional[KnowledgeGraph]:
        """
        Extracts a KnowledgeGraph from a given MarketEvent.

        Args:
            event: The MarketEvent to process.

        Returns:
            A KnowledgeGraph object, or None if extraction fails.
        """
        try:
            # 1. Render the prompt
            full_prompt = self.prompt_renderer.render(
                template_content=self.prompt_template,
                event=event.dict() # Pass event as a dict
            )
            
            # 2. Execute the LLM call
            response_json = await self.gemini_pool.generate_json(
                model_name="gemini-1.5-flash", # Use a fast model for extraction
                prompt=full_prompt,
                system_prompt="You are a data extraction expert. Respond ONLY with the requested JSON schema for a knowledge graph."
            )
            
            if not response_json or 'nodes' not in response_json or 'relations' not in response_json:
                logger.warning(f"Invalid JSON response from RelationExtractor for event {event.event_id}")
                return None

            # 3. Validate and build the KnowledgeGraph
            kg = KnowledgeGraph(
                nodes=[KGNode(**node) for node in response_json.get('nodes', [])],
                relations=[KGRelation(**rel) for rel in response_json.get('relations', [])]
            )
            
            logger.info(f"Successfully extracted {len(kg.nodes)} nodes and {len(kg.relations)} relations from event {event.event_id}.")
            return kg

        except Exception as e:
            logger.error(f"Error during relation extraction for event {event.event_id}: {e}", exc_info=True)
            return None

"""
Phoenix_project/ai/relation_extractor.py
[Phase 5 Task 3] Fix PromptManager API Call.
Use PromptRenderer for context injection instead of passing kwargs to get_prompt.
"""
from typing import Dict, Any, List
import json
from Phoenix_project.api.gateway import APIGateway
from Phoenix_project.ai.prompt_manager import PromptManager
# [Task 3] Import PromptRenderer
from Phoenix_project.ai.prompt_renderer import PromptRenderer
from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

class RelationExtractor:
    """
    Identical to GraphEncoder. Uses an LLM to extract entities and relationships
    from text to build a Knowledge Graph.
    """

    def __init__(self, api_gateway: APIGateway, prompt_manager: PromptManager):
        self.api_gateway = api_gateway
        self.prompt_manager = prompt_manager
        # [Task 3] Initialize PromptRenderer
        self.prompt_renderer = PromptRenderer(prompt_manager)
        self.prompt_name = "extract_relations" 
        logger.info("RelationExtractor initialized.")

    async def extract_relations(self, text_content: str, metadata: Dict[str, Any] = None) -> Dict[str, List[Dict]]:
        """
        Takes a block of text and converts it into a list of structured
        graph relations (nodes and edges).
        """
        if not text_content:
            logger.warning("RelationExtractor received empty text content.")
            return {"nodes": [], "edges": []}
            
        # [Task 3] Fix: Use prompt_renderer.render instead of get_prompt with args
        try:
            context = {
                "text_content": text_content,
                "metadata": metadata or {}
            }
            prompt = self.prompt_renderer.render(self.prompt_name, context)
        except Exception as e:
            logger.error(f"Error rendering prompt '{self.prompt_name}': {e}")
            return {"nodes": [], "edges": []}
        
        if not prompt:
            logger.error(f"Rendered prompt '{self.prompt_name}' is empty.")
            return {"nodes": [], "edges": []}

        try:
            # We explicitly ask the LLM to return JSON
            raw_response = await self.api_gateway.send_request(
                model_name="gemini-pro", 
                prompt=prompt,
                temperature=0.1, 
                max_tokens=2048,
            )
            
            return self._parse_graph_response(raw_response)
            
        except Exception as e:
            logger.error(f"Error extracting relations: {e}", exc_info=True)
            return {"nodes": [], "edges": []}

    def _parse_graph_response(self, response: str) -> Dict[str, List[Dict]]:
        """
        Robustly parses the LLM's string response, expecting JSON.
        """
        try:
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()
                
            graph_data = json.loads(json_str)
            
            if "nodes" in graph_data and "edges" in graph_data and \
               isinstance(graph_data["nodes"], list) and \
               isinstance(graph_data["edges"], list):
                logger.info(f"Successfully parsed relations: {len(graph_data['nodes'])} nodes, {len(graph_data['edges'])} edges.")
                return graph_data
            else:
                logger.warning(f"Parsed JSON lacks required 'nodes'/'edges' keys or valid types: {json_str[:100]}...")
                return {"nodes": [], "edges": []}
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from relation response: {e}. Response: {response[:200]}...")
            return {"nodes": [], "edges": []}
        except Exception as e:
            logger.error(f"Error parsing relation response: {e}", exc_info=True)
            return {"nodes": [], "edges": []}

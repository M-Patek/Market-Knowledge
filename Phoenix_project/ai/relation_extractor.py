from typing import Dict, Any, List
import json
from api.gateway import APIGateway
from ai.prompt_manager import PromptManager
from monitor.logging import get_logger

logger = get_logger(__name__)

class RelationExtractor:
    """
    Identical to GraphEncoder. Uses an LLM to extract entities and relationships
    from text to build a Knowledge Graph.
    
    This class is aliased for semantic clarity in different parts of the system.
    """

    def __init__(self, api_gateway: APIGateway, prompt_manager: PromptManager):
        self.api_gateway = api_gateway
        self.prompt_manager = prompt_manager
        self.prompt_name = "extract_relations" # Prompt optimized for (Subject, Predicate, Object)
        logger.info("RelationExtractor initialized.")

    async def extract_relations(self, text_content: str, metadata: Dict[str, Any] = None) -> Dict[str, List[Dict]]:
        """
        Takes a block of text and converts it into a list of structured
        graph relations (nodes and edges).
        
        Args:
            text_content (str): The input text (e.g., news article, agent reasoning).
            metadata (Dict[str, Any]): Optional metadata about the text (source, timestamp).
            
        Returns:
            Dict[str, List[Dict]]: A dictionary with "nodes" and "edges" keys.
                                  e.g., {"nodes": [...], "edges": [...]}
        """
        if not text_content:
            logger.warning("RelationExtractor received empty text content.")
            return {"nodes": [], "edges": []}
            
        prompt = self.prompt_manager.get_prompt(
            self.prompt_name,
            text_content=text_content,
            metadata=metadata or {}
        )
        
        if not prompt:
            logger.error(f"Could not get prompt '{self.prompt_name}'.")
            return {"nodes": [], "edges": []}

        try:
            # We explicitly ask the LLM to return JSON
            raw_response = await self.api_gateway.send_request(
                model_name="gemini-pro", # Use a model good at JSON output
                prompt=prompt,
                temperature=0.1, # Low temperature for structured output
                max_tokens=2048,
            )
            
            return self._parse_graph_response(raw_response)
            
        except Exception as e:
            logger.error(f"Error extracting relations: {e}", exc_info=True)
            return {"nodes": [], "edges": []}

    def _parse_graph_response(self, response: str) -> Dict[str, List[Dict]]:
        """
        Robustly parses the LLM's string response, expecting JSON.
        (This is identical to GraphEncoder's parser)
        """
        try:
            # The LLM response might be wrapped in ```json ... ```
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            else:
                json_str = response.strip()
                
            graph_data = json.loads(json_str)
            
            # Validate basic structure
            if "nodes" in graph_data and "edges" in graph_data and \
               isinstance(graph_data["nodes"], list) and \
               isinstance(graph_data["edges"], list):
                logger.info(f"Successfully parsed relations: {len(graph_data['nodes'])} nodes, {len(graph_data['edges'])} edges.")
                return graph_data
            else:
                logger.warning(f"Parsed JSON lacks required 'nodes'/'edges' keys or valid types: {json_str}")
                return {"nodes": [], "edges": []}
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON from relation response: {e}. Response: {response[:200]}...")
            return {"nodes": [], "edges": []}
        except Exception as e:
            logger.error(f"Error parsing relation response: {e}", exc_info=True)
            return {"nodes": [], "edges": []}

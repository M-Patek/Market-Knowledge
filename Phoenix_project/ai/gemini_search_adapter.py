# Phoenix_project/ai/gemini_search_adapter.py

import logging
from typing import List, Dict, Any, Optional
from google.ai.generativelanguage_v1beta.types import (
    Tool,
    GoogleSearchRetrieval,
    DynamicRetrievalConfig
)
from Phoenix_project.api.gemini_pool_manager import GeminiPoolManager

logger = logging.getLogger(__name__)

class GeminiSearchAdapter:
    """
    Adapts Gemini's Grounding with Google Search to match the interface
    expected by the Retriever (formerly TavilyClient).
    Utilizes the existing unlimited Gemini key pool.
    """
    def __init__(self, gemini_manager: GeminiPoolManager):
        self.gemini_manager = gemini_manager
        # Configure the tool to force Google Search
        # We set dynamic_threshold to 0.0 to ensure search is triggered for every query
        self._tool = Tool(
            google_search_retrieval=GoogleSearchRetrieval(
                dynamic_retrieval_config=DynamicRetrievalConfig(
                    mode=DynamicRetrievalConfig.Mode.MODE_DYNAMIC,
                    dynamic_threshold=0.0 
                )
            )
        )

    async def search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Performs a Google Search using Gemini Grounding.
        
        Args:
            query (str): The search query.
            max_results (int): The maximum number of results to return (approximate, depends on Grounding output).
            
        Returns:
            Dict[str, Any]: A dictionary containing 'results', matching the Tavily format:
                            {'results': [{'title':..., 'url':..., 'content':...}]}
        """
        # Construct a prompt to induce searching and summarization
        prompt = f"Search for latest details on: '{query}'. Summarize key facts."
        
        try:
            # We use 'gemini-1.5-flash' for speed and efficiency, as it supports Grounding
            async with self.gemini_manager.get_client("gemini-1.5-flash") as client:
                response = await client.generate_content_async(
                    contents=[prompt],
                    tools=[self._tool]
                )
            
            results = []
            
            # Extract grounding metadata (parsed by our updated GeminiPoolManager)
            metadata = response.get("grounding_metadata")
            chunks = getattr(metadata, 'grounding_chunks', []) if metadata else []
            
            # Strategy: Convert Gemini's chunks into the expected list format
            for chunk in chunks:
                web = getattr(chunk, 'web', None)
                if web:
                    results.append({
                        "title": getattr(web, 'title', 'Gemini Search Result'),
                        "url": getattr(web, 'uri', ''),
                        # Use retrieved_context if available, else fallback to a snippet of the AI's answer
                        "content": getattr(chunk, 'retrieved_context', None) or response.get("text", "")[:200]
                    })
                    
            # Fallback/Supplement: If search was conducted but no chunks were parsed (rare), 
            # or to ensure the AI's synthesis is included.
            response_text = response.get("text", "")
            if response_text:
                # If we have very few results, or just want to ensure the summary is present
                if not results:
                     results.append({
                        "title": "Gemini AI Summary",
                        "url": "ai://gemini_summary",
                        "content": response_text
                    })
            
            logger.info(f"Gemini Search found {len(results)} results/sources for '{query}'")
            return {"results": results[:max_results]}

        except Exception as e:
            logger.error(f"Gemini Search Adapter failed for query '{query}': {e}", exc_info=True)
            # Return empty list on failure to prevent crashing the Retriever
            return {"results": []}

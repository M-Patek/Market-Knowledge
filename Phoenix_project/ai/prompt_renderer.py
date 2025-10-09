# ai/prompt_renderer.py
"""
Handles the rendering of AI prompts from JSON templates and context data.
This decouples the prompt engineering logic from the AI client logic.
"""
import json
import logging
from typing import List, Dict, Any

def render_prompt(template_path: str, ticker: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Loads a JSON prompt template, injects context, and returns the final prompt string
    along with metadata used for validation.

    Args:
        template_path: Path to the .json prompt template file.
        ticker: The asset ticker being analyzed.
        retrieved_docs: A list of document dictionaries from the retrieval system.

    Returns:
        A dictionary containing the 'final_prompt' string and 'meta' data.
    """
    logger = logging.getLogger("PhoenixProject.PromptRenderer")
    
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            template_data = json.load(f)

        # 1. Format the retrieved documents into a string for the prompt context.
        # This part is crucial for the AI to understand its knowledge base.
        context_str_parts = []
        retrieved_doc_ids = []
        for doc in retrieved_docs:
            doc_id = doc.get('id', 'N/A')
            retrieved_doc_ids.append(doc_id)
            content_snippet = doc.get('content', '')[:500] # Truncate for brevity
            context_str_parts.append(f"START_DOC (ID: {doc_id})\n{content_snippet}\nEND_DOC")
        
        context_str = "\n---\n".join(context_str_parts)
        if not context_str:
            context_str = "No documents were retrieved for this asset."

        # 2. Render the final prompt by filling placeholders in the template.
        # We use a simple .format() here, but a more complex engine like Jinja2 could be used.
        full_prompt_structure = {
            "role": template_data.get("role"),
            "task": template_data.get("task"),
            "rules": template_data.get("rules"),
            "output_format": template_data.get("output_format"),
            "context": context_str
        }
        
        final_prompt_str = json.dumps(full_prompt_structure, indent=2)

        return {
            "final_prompt": final_prompt_str,
            "meta": {
                "template_path": template_path,
                "retrieved_doc_ids": retrieved_doc_ids,
                "ticker": ticker
            }
        }

    except FileNotFoundError:
        logger.error(f"Prompt template not found at path: {template_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to render prompt from template '{template_path}': {e}")
        raise

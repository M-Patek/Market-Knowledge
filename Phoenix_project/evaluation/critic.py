from typing import Dict, Any, Optional
from ai.prompt_manager import PromptManager
from api.gateway import APIGateway
from monitor.logging import get_logger

logger = get_logger(__name__)

class Critic:
    """
    A generic evaluation agent that critiques a given piece of text
    (e.g., reasoning, a plan, a news summary) based on a set of criteria.
    """

    def __init__(self, api_gateway: APIGateway, prompt_manager: PromptManager):
        self.api_gateway = api_gateway
        self.prompt_manager = prompt_manager
        self.prompt_name = "critique_reasoning" # Default prompt
        logger.info("Critic initialized.")

    async def critique(
        self,
        text_to_critique: str,
        context: Optional[str] = None,
        criteria: Optional[str] = None,
        prompt_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Performs a critique.
        
        Args:
            text_to_critique (str): The text to be evaluated.
            context (Optional[str]): Supporting context for the critique.
            criteria (Optional[str]): Specific criteria to focus on. If None,
                                      prompt default is used.
            prompt_name (Optional[str]): The prompt template to use.
            
        Returns:
            A structured critique (ideally JSON).
        """
        
        prompt_name = prompt_name or self.prompt_name
        
        prompt = self.prompt_manager.get_prompt(
            prompt_name,
            text_to_critique=text_to_critique,
            context=context or "No context provided.",
            criteria=criteria or "Default criteria: logical consistency, bias, and completeness."
        )
        
        if not prompt:
            logger.error(f"Could not get prompt '{prompt_name}'.")
            return {"error": f"Prompt '{prompt_name}' not found."}

        try:
            raw_response = await self.api_gateway.send_request(
                model_name="gemini-pro", # Or gemini-1.5-pro for complex critiques
                prompt=prompt,
                temperature=0.2,
                max_tokens=1024
            )
            
            # Assume prompt asks for JSON
            return self._parse_critique_response(raw_response)
            
        except Exception as e:
            logger.error(f"Error during critique: {e}", exc_info=True)
            return {"error": f"LLM API failed: {e}"}

    def _parse_critique_response(self, response: str) -> Dict[str, Any]:
        """
        Parses the raw response from the LLM, expecting JSON.
        """
        try:
            import json
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            else:
                json_str = response.strip()
                
            parsed_data = json.loads(json_str)
            return parsed_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON critique: {e}. Response: {response[:200]}...")
            return {"critique_text": response, "error": "JSON parse failed"}
        except Exception as e:
            logger.error(f"Error parsing critique response: {e}", exc_info=True)
            return {"error": f"Parsing failed: {e}"}

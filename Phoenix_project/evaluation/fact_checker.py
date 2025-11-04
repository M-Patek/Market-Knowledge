from typing import Dict, Any, Optional
import json
from Phoenix_project.ai.prompt_manager import PromptManager
from Phoenix_project.api.gateway import APIGateway
from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

class FactChecker:
    """
    Uses an LLM with search capabilities (e.g., Gemini with Google Search tool)
    to verify factual claims made in a piece of reasoning.
    
    NOTE: This requires the APIGateway to support enabling tools
    like Google Search, which is not explicitly implemented in the
    provided `APIGateway` or `GeminiPoolManager`. This implementation
    *assumes* it can be enabled via `send_request`.
    """

    def __init__(self, api_gateway: APIGateway, prompt_manager: PromptManager):
        self.api_gateway = api_gateway
        self.prompt_manager = prompt_manager
        self.prompt_name = "fact_check_reasoning"
        logger.info("FactChecker initialized.")

    async def check_facts(self, reasoning_text: str) -> Dict[str, Any]:
        """
        Checks the factual claims in the provided reasoning.
        
        Args:
            reasoning_text (str): The AI's reasoning to be checked.
            
        Returns:
            A structured report, e.g.:
            {
                "overall_support": "Supported" | "Refuted" | "No Information",
                "checked_claims": [
                    {"claim": "...", "support": "...", "evidence": "..."}
                ]
            }
        """
        
        prompt = self.prompt_manager.get_prompt(
            self.prompt_name,
            reasoning_text=reasoning_text
        )
        
        if not prompt:
            logger.error(f"Could not get prompt '{self.prompt_name}'.")
            return self._error_response("Prompt missing")

        try:
            # This is the critical part. We need to tell the gateway
            # to enable Google Search. This is a hypothetical extension
            # of the send_request method.
            # A more realistic `send_request` would take `tools` as an arg.
            
            # --- HACK: Prepend prompt to instruct model to use search ---
            # This is less reliable than the API `tools` parameter.
            search_prompt = (
                "You MUST use Google Search to verify the factual claims in the following text. "
                "Do not use your internal knowledge. Provide sources for your checks.\n\n"
                + prompt
            )
            
            logger.warning("FactChecker is using a prompt-based search instruction. "
                           "This is less reliable than enabling the API `tools` parameter.")

            raw_response = await self.api_gateway.send_request(
                model_name="gemini-1.5-pro", # Model that supports search
                prompt=search_prompt,
                temperature=0.0, # Be factual
                max_tokens=2048
            )
            
            return self._parse_fact_check_response(raw_response)
            
        except Exception as e:
            logger.error(f"Error during fact check: {e}", exc_info=True)
            return self._error_response(f"LLM API failed: {e}")

    def _parse_fact_check_response(self, response: str) -> Dict[str, Any]:
        """
        Parses the LLM response, which should be structured (JSON).
        """
        try:
            import json
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            else:
                json_str = response.strip()
                
            parsed_data = json.loads(json_str)
            
            if "overall_support" not in parsed_data or "checked_claims" not in parsed_data:
                logger.warning(f"Fact check response missing required keys: {response[:100]}...")
                return self._error_response("Parsed JSON missing required keys.")
                
            logger.info(f"Fact check complete. Overall support: {parsed_data['overall_support']}")
            return parsed_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON fact check: {e}. Response: {response[:200]}...")
            return self._error_response(f"JSON parse failed: {e}")
        except Exception as e:
            logger.error(f"Error parsing fact check response: {e}", exc_info=True)
            return self._error_response(f"Parsing failed: {e}")

    def _error_response(self, error_msg: str) -> Dict[str, Any]:
        """Returns a standardized error dictionary."""
        return {
            "overall_support": "Unknown",
            "checked_claims": [],
            "error": error_msg
        }

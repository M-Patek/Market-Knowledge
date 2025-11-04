from typing import Dict, Any, Optional
from Phoenix_project.api.gateway import APIGateway
from Phoenix_project.ai.prompt_manager import PromptManager
from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

class MarketStatePredictor:
    """
    Uses an LLM to predict the market state (e.g., "Bullish", "Bearish", "Volatile")
    based on the current context.
    
    This is a specialized agent within the reasoning ensemble.
    """

    def __init__(self, api_gateway: APIGateway, prompt_manager: PromptManager):
        self.api_gateway = api_gateway
        self.prompt_manager = prompt_manager
        self.prompt_name = "predict_market_state"
        logger.info("MarketStatePredictor initialized.")

    async def predict_state(self, context: str) -> Dict[str, Any]:
        """
        Predicts the market state and provides reasoning.
        
        Args:
            context (str): The formatted context string (news, market data, etc.).
            
        Returns:
            Dict[str, Any]: e.g., {"state": "Volatile", "confidence": 0.8, "reasoning": "..."}
        """
        prompt = self.prompt_manager.get_prompt(
            self.prompt_name,
            context=context
        )
        
        if not prompt:
            logger.error(f"Could not get prompt '{self.prompt_name}'.")
            return self._error_response("Prompt missing")

        try:
            raw_response = await self.api_gateway.send_request(
                model_name="gemini-pro", # Or a model fine-tuned for this
                prompt=prompt,
                temperature=0.3,
                max_tokens=500
            )
            
            return self._parse_prediction_response(raw_response)
            
        except Exception as e:
            logger.error(f"Error predicting market state: {e}", exc_info=True)
            return self._error_response(str(e))

    def _parse_prediction_response(self, response: str) -> Dict[str, Any]:
        """
        Parses the raw LLM response into a structured state prediction.
        
        Assumes the prompt guides the LLM to return JSON or a simple format:
        STATE: [STATE]
        CONFIDENCE: [0.0-1.0]
        REASONING: [Text]
        """
        try:
            # Simple key-value parsing (less robust than JSON)
            state = "Neutral"
            confidence = 0.5
            reasoning = response # Default

            lines = response.split('\n')
            for line in lines:
                if line.upper().startswith("STATE:"):
                    state = line.split(":", 1)[1].strip()
                elif line.upper().startswith("CONFIDENCE:"):
                    confidence = float(line.split(":", 1)[1].strip())
                elif line.upper().startswith("REASONING:"):
                    reasoning = line.split(":", 1)[1].strip()
            
            # Basic validation
            if state not in ["Bullish", "Bearish", "Volatile", "Neutral", "Sideways"]:
                logger.warning(f"Parsed unknown state: {state}.Defaulting to Neutral.")
                state = "Neutral"
                
            confidence = max(0.0, min(1.0, confidence))

            logger.info(f"Predicted market state: {state} (Confidence: {confidence:.2f})")
            return {
                "state": state,
                "confidence": confidence,
                "reasoning": reasoning
            }

        except Exception as e:
            logger.error(f"Failed to parse prediction response: {e}. Response: {response[:100]}...")
            return self._error_response(f"Parsing failed: {e}")

    def _error_response(self, error_msg: str) -> Dict[str, Any]:
        """Returns a standardized error dictionary."""
        return {
            "state": "Unknown",
            "confidence": 0.0,
            "reasoning": f"Failed to predict state: {error_msg}"
        }

from typing import Dict, Any, Optional
from urllib.parse import urlparse
from Phoenix_project.api.gateway import APIGateway
from Phoenix_project.ai.prompt_manager import PromptManager
from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

class SourceCredibility:
    """
    Assesses the credibility of an information source (e.g., news website)
    using both a predefined list and dynamic LLM analysis.
    """

    def __init__(self, api_gateway: APIGateway, prompt_manager: PromptManager, config: Dict[str, Any]):
        self.api_gateway = api_gateway
        self.prompt_manager = prompt_manager
        
        # Predefined list of trusted/untrusted sources
        # Example: {"bloomberg.com": 0.9, "randomblog.com": 0.2}
        self.known_sources: Dict[str, float] = config.get("known_sources", {})
        
        self.prompt_name = "assess_source_credibility"
        logger.info(f"SourceCredibility initialized with {len(self.known_sources)} known sources.")

    def _get_domain(self, source_url: str) -> Optional[str]:
        """Extracts the netloc (domain) from a URL."""
        try:
            parsed = urlparse(source_url)
            return parsed.netloc.replace("www.", "")
        except Exception:
            return None

    async def assess_credibility(self, source_name: str, source_url: Optional[str] = None, article_snippet: Optional[str] = None) -> Dict[str, Any]:
        """
        Assesses the credibility of a source.
        
        Args:
            source_name (str): The name of the source (e.g., "Bloomberg", "ZeroHedge").
            source_url (Optional[str]): The URL of the source or article.
            article_snippet (Optional[str]): A snippet of text to help the LLM assess bias.

        Returns:
            Dict[str, Any]: e.g., {"score": 0.85, "rating": "High", "reasoning": "..."}
        """
        
        # 1. Check predefined list
        domain = None
        if source_url:
            domain = self._get_domain(source_url)
            if domain and domain in self.known_sources:
                score = self.known_sources[domain]
                rating = self._score_to_rating(score)
                logger.info(f"Found known source '{domain}'. Score: {score}")
                return {
                    "score": score,
                    "rating": rating,
                    "reasoning": f"Source '{domain}' is on the predefined list with a score of {score}.",
                    "source": "Predefined List"
                }

        # 2. If not in list, use LLM to assess
        logger.info(f"Source '{source_name}' (Domain: {domain}) not in known list. Assessing with LLM.")
        
        prompt = self.prompt_manager.get_prompt(
            self.prompt_name,
            source_name=source_name,
            source_url=source_url or "N/A",
            article_snippet=article_snippet or "N/A"
        )
        
        if not prompt:
            logger.error(f"Could not get prompt '{self.prompt_name}'.")
            return self._error_response("Prompt missing")

        try:
            raw_response = await self.api_gateway.send_request(
                model_name="gemini-pro",
                prompt=prompt,
                temperature=0.2,
                max_tokens=300
            )
            
            return self._parse_assessment_response(raw_response, source_name)
            
        except Exception as e:
            logger.error(f"Error assessing credibility for '{source_name}': {e}", exc_info=True)
            return self._error_response(str(e))

    def _score_to_rating(self, score: float) -> str:
        """Converts a numeric score (0-1) to a rating."""
        if score >= 0.8: return "High"
        if score >= 0.6: return "Moderate-High"
        if score >= 0.4: return "Moderate-Low"
        if score >= 0.2: return "Low"
        return "Very Low"

    def _parse_assessment_response(self, response: str, source_name: str) -> Dict[str, Any]:
        """
        Parses the LLM response.
        
        Assumes format:
        SCORE: [0.0-1.0]
        RATING: [RATING]
        REASONING: [Text]
        """
        try:
            score = 0.5 # Default
            rating = "Moderate-Low"
            reasoning = response

            lines = response.split('\n')
            for line in lines:
                if line.upper().startswith("SCORE:"):
                    score = float(line.split(":", 1)[1].strip())
                elif line.upper().startswith("RATING:"):
                    rating = line.split(":", 1)[1].strip()
                elif line.upper().startswith("REASONING:"):
                    reasoning = line.split(":", 1)[1].strip()
            
            score = max(0.0, min(1.0, score))
            
            logger.info(f"LLM assessment for '{source_name}': Score {score} ({rating})")
            return {
                "score": score,
                "rating": rating,
                "reasoning": reasoning,
                "source": "LLM Assessment"
            }

        except Exception as e:
            logger.error(f"Failed to parse assessment response: {e}. Response: {response[:100]}...")
            return self._error_response(f"Parsing failed: {e}")

    def _error_response(self, error_msg: str) -> Dict[str, Any]:
        """Returns a standardized error dictionary."""
        return {
            "score": 0.5,
            "rating": "Unknown",
            "reasoning": f"Failed to assess credibility: {error_msg}",
            "source": "Error"
        }

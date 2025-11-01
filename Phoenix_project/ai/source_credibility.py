import json
from typing import Dict, Any, List
from urllib.parse import urlparse
from ..monitor.logging import get_logger

logger = get_logger(__name__)

class SourceCredibilityModel:
    """
    Assesses the credibility of information sources based on predefined scores.
    This acts as a simple, rule-based model.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the model with credibility scores from the config.
        
        Args:
            config (Dict[str, Any]): Expects a 'source_credibility' key containing
                                      a dictionary mapping domain names to scores (0.0 - 1.0).
        """
        self.credibility_scores = config.get('source_credibility', {})
        if not self.credibility_scores:
            logger.warning("Source credibility scores are not defined in config. Using defaults.")
            self.credibility_scores = {
                "default": 0.5,
                "reuters.com": 0.9,
                "bloomberg.com": 0.9,
                "wsj.com": 0.85,
                "sec.gov": 0.95,
                "benzinga.com": 0.75,
                "finance.yahoo.com": 0.7,
                "twitter.com": 0.4, # Highly variable, depends on user
                "seekingalpha.com": 0.6,
                "default_structured_db": 0.9, # Internal DBs are generally trusted
                "default_temporal_db": 0.8,
            }
        
        logger.info(f"SourceCredibilityModel initialized with {len(self.credibility_scores)} sources.")

    def score_sources(self, context_bundle: Dict[str, List[Dict[str, Any]]]) -> Dict[str, float]:
        """
        Scores all sources found in the RAG context bundle.
        
        Args:
            context_bundle (Dict[str, List[Dict[str, Any]]]): The output from the Retriever,
                e.g., {"vector_context": [...], "temporal_context": [...], ...}
        
        Returns:
            Dict[str, float]: A dictionary mapping unique source identifiers (like
                              event_id or domain) to a credibility score.
        """
        scores = {}
        
        # Score Vector Context (e.g., news articles)
        for item in context_bundle.get('vector_context', []):
            try:
                source_domain = self._extract_domain(item.get('metadata', {}).get('url'))
                score = self.credibility_scores.get(source_domain, self.credibility_scores.get('default', 0.5))
                # Use event_id as the unique key for this piece of context
                scores[item.get('id', source_domain)] = score
            except Exception as e:
                logger.warning(f"Failed to score vector context item {item.get('id')}: {e}")

        # Score Temporal Context (e.g., past events)
        for item in context_bundle.get('temporal_context', []):
            try:
                source_id = item.get('source', 'default_temporal_db')
                score = self.credibility_scores.get(source_id, self.credibility_scores.get('default', 0.5))
                scores[item.get('id', source_id)] = score
            except Exception as e:
                logger.warning(f"Failed to score temporal context item {item.get('id')}: {e}")

        # Score Structured Context (e.g., financial data)
        for item in context_bundle.get('structured_context', []):
            try:
                source_id = item.get('source', 'default_structured_db')
                score = self.credibility_scores.get(source_id, self.credibility_scores.get('default_structured_db', 0.9))
                scores[item.get('id', source_id)] = score
            except Exception as e:
                logger.warning(f"Failed to score structured context item {item.get('id')}: {e}")
                
        return scores

    def _extract_domain(self, url: str) -> str:
        """
        Utility to extract the netloc (domain) from a URL.
        
        Examples:
            "https.api.benzinga.com/..." -> "benzinga.com"
            "www.wsj.com/articles/..." -> "wsj.com"
        """
        if not url:
            return "unknown"
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            if not domain:
                return "unknown"
            
            # Clean common subdomains like 'www' or 'api'
            parts = domain.split('.')
            if len(parts) > 2 and parts[0] in ['www', 'api', 'finance']:
                return '.'.join(parts[1:])
            
            return domain
            
        except Exception as e:
            logger.debug(f"Could not parse domain from URL '{url}': {e}")
            return "unknown"

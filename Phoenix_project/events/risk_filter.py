import yaml
from typing import Dict, Any, Optional

from ..monitor.logging import get_logger
from ..core.schemas.data_schema import MarketEvent

logger = get_logger(__name__)

class RiskFilter:
    """
    A high-speed, synchronous filter that performs a "first pass"
    on incoming events to check for systemic risk keywords.
    
    Its purpose is to *immediately* flag high-importance events
    (e.g., "FED", "WAR", "CRISIS") before they even enter the
    main processing queue.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the RiskFilter and loads the keyword configuration.
        
        Args:
            config: The main system configuration.
        """
        self.config = config.get('risk_filter', {})
        self.filter_config_path = self.config.get('config_path', 'config/event_filter_config.yaml')
        self.keywords = self._load_keywords()
        
        logger.info(f"RiskFilter initialized. Loaded {len(self.keywords)} high-risk keywords.")

    def _load_keywords(self) -> Dict[str, Dict]:
        """
        Loads keyword rules from the event_filter_config.yaml file.
        
        Expected format in YAML:
        keywords:
          systemic:
            - "fed rate"
            - "ecb"
          geopolitical:
            - "war"
        """
        try:
            # Construct path relative to this file
            import os
            config_path = os.path.join(
                os.path.dirname(__file__), 
                '..', 
                self.filter_config_path
            )
            
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # We flatten the keywords into a single set for fast lookup
            all_keywords = {}
            for category, keyword_list in config_data.get('keywords', {}).items():
                for keyword in keyword_list:
                    # Store as lowercase for case-insensitive matching
                    all_keywords[keyword.lower()] = {"category": category}
                    
            return all_keywords
            
        except FileNotFoundError:
            logger.error(f"RiskFilter config file not found at: {config_path}")
            return {}
        except Exception as e:
            logger.error(f"Failed to load RiskFilter keywords from {config_path}: {e}", exc_info=True)
            return {}

    def check_event(self, event: MarketEvent) -> Optional[Dict[str, Any]]:
        """
        Synchronously checks if an event's text contains any high-risk keywords.
        This must be *very* fast.
        
        Args:
            event (MarketEvent): The event to check.
            
        Returns:
            Optional[Dict]: A dict with match info if found (e.g., 
                            {"keyword": "fed rate", "category": "systemic"}),
                            otherwise None.
        """
        
        # Combine headline and summary for searching
        text_to_search = (event.headline + " " + event.summary).lower()
        
        if not text_to_search:
            return None

        # This is a simple O(N*M) search.
        # For performance, a real system would use Aho-Corasick or
        # another multi-pattern matching algorithm.
        for keyword, info in self.keywords.items():
            if keyword in text_to_search:
                logger.warning(f"HIGH-RISK EVENT DETECTED (Event: {event.event_id}). "
                               f"Keyword: '{keyword}', Category: {info['category']}")
                
                return {
                    "keyword_match": keyword,
                    "category": info['category']
                }
                
        # No match
        return None

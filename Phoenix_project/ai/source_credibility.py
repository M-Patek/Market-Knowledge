# Placeholder for source credibility logic
_default_store = {}

# Configure logger for this module (Layer 12)
from observability import get_logger
logger = get_logger(__name__)

class SourceCredibilityStore:
    """
    Manages the credibility scores for different data sources.
    """

    def __init__(self, store: dict = None):
        # Using a default global store for simulation simplicity
        self.store = _default_store

    def update_source_credibility(self, source_type: str, source_name: str, value: float):
        logger.info(f"SourceCredibility: Updating {source_type}:{source_name} to {value}")
        
        if source_type not in self.store:
            self.store[source_type] = {}
        
        self.store[source_type][source_name] = value

    def get_credibility(self, source_type: str, source_name: str) -> float:
        """Retrieves the credibility for a source, defaulting to 1.0."""
        return self.store.get(source_type, {}).get(source_name, 1.0)

"""
Client for handling text, table, and image embeddings (Layer 13).
"""

from observability import get_logger

# Configure logger for this module (Layer 12)
logger = get_logger(__name__)

class EmbeddingClient:
    """
    Provides an interface to various embedding models.
    """

    def embed_text(self, text: str) -> list[float]:
        """Generates an embedding for a given text."""
        logger.info("Generating text embedding (mock).")
        return [0.1] * 768 # Return a mock vector of a common size

    def embed_table(self, table_data: dict) -> list[float]:
        """Generates an embedding for tabular data."""
        logger.info("Generating table embedding (mock).")
        return [0.2] * 768 # Return a mock vector

    def embed_image(self, image_path: str) -> list[float]:
        """Generates an embedding for an image."""
        logger.info(f"Generating image embedding for {image_path} (mock).")
        return [0.3] * 768 # Return a mock vector

from typing import List, Dict, Any, Optional
import asyncio
from Phoenix_project.models.registry import ModelRegistry
from Phoenix_project.api.gateway import APIGateway
from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

class EmbeddingClient:
    """
    Handles the generation of text embeddings using a specified model.
    Manages batching and API calls.
    """

    def __init__(self, model_registry: ModelRegistry, api_gateway: APIGateway, default_model_id: str = "text-embedding-3-large"):
        self.model_registry = model_registry
        self.api_gateway = api_gateway
        self.default_model_id = default_model_id
        self.model_config = self.model_registry.get_model_config(self.default_model_id)
        
        if not self.model_config:
            logger.error(f"Default embedding model '{self.default_model_id}' not found in registry.")
            raise ValueError(f"Embedding model '{self.default_model_id}' configuration not found.")
            
        logger.info(f"EmbeddingClient initialized with model: {self.default_model_id}")

    async def get_embedding(self, text: str, model_id: Optional[str] = None) -> List[float]:
        """
        Generates an embedding for a single string of text.
        
        Returns:
            List[float]: The embedding vector.
        """
        model_id = model_id or self.default_model_id
        try:
            # Use batch endpoint even for single text for consistency
            embeddings = await self.get_embeddings_batch([text], model_id)
            return embeddings[0]
        except Exception as e:
            logger.error(f"Error generating single embedding: {e}", exc_info=True)
            return []

    async def get_embeddings_batch(self, texts: List[str], model_id: Optional[str] = None) -> List[List[float]]:
        """
        Generates embeddings for a batch of texts.
        
        Args:
            texts (List[str]): A list of strings to embed.
            model_id (Optional[str]): Override the default embedding model.
            
        Returns:
            List[List[float]]: A list of embedding vectors.
        """
        if not texts:
            return []
            
        model_id = model_id or self.default_model_id
        model_config = self.model_registry.get_model_config(model_id)
        
        if not model_config:
            logger.error(f"Embedding model '{model_id}' not found in registry.")
            return [[] for _ in texts]
            
        logger.info(f"Generating {len(texts)} embeddings using model '{model_id}'...")

        # The API gateway should ideally handle the actual batching logic
        # if the underlying API provider (e.g., OpenAI) supports it.
        # Here, we assume the gateway's `send_embedding_request`
        # can take a list of texts.

        try:
            # This is a hypothetical method on APIGateway
            results = await self.api_gateway.send_embedding_request(
                model_name=model_config.provider_model_id,
                texts=texts,
                dimensions=model_config.dimensions
            )
            
            # Assuming `results` is a list of embedding vectors
            if not results or len(results) != len(texts):
                logger.warning(f"Embedding API returned mismatched results. Expected {len(texts)}, got {len(results or [])}")
                return [[] for _ in texts]

            logger.info(f"Successfully generated {len(results)} embeddings.")
            return results
            
        except Exception as e:
            logger.error(f"Error generating embeddings batch: {e}", exc_info=True)
            return [[] for _ in texts]

    def get_model_dimensions(self) -> int:
        """Returns the embedding dimensions of the default model."""
        return self.model_config.dimensions

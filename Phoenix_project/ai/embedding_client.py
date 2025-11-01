import os
import json
import numpy as np
from openai import OpenAI
from typing import List, Dict, Any, Optional
from ..monitor.logging import get_logger
from ..storage.s3_client import S3Client

logger = get_logger(__name__)

class EmbeddingClient:
    """
    Manages the creation, retrieval, and caching of text embeddings.
    Supports OpenAI's embedding models and caches results to S3.
    """

    def __init__(self, config: Dict[str, Any], s3_client: S3Client = None):
        """
        Initializes the EmbeddingClient.

        Args:
            config (Dict[str, Any]): Configuration dictionary, expects 'embedding_model_id'
                                      and 'embedding_model_version'.
            s3_client (S3Client, optional): S3 client for caching. Defaults to None.
        """
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            logger.error("OPENAI_API_KEY environment variable not set.")
            raise ValueError("OPENAI_API_KEY not set")
            
        self.client = OpenAI(api_key=self.api_key)
        self.model_id = config.get('embedding_model_id', 'text-embedding-3-large')
        self.model_version = config.get('embedding_model_version', 'v1')
        self.model_config = self._load_model_config()
        self.dimensions = self.model_config.get('dimensions')
        
        self.s3_client = s3_client
        self.s3_bucket = config.get('s3_cache_bucket')
        self.cache_prefix = f"embeddings/{self.model_id}_{self.model_version}/"

        logger.info(f"EmbeddingClient initialized with model: {self.model_id} (Version: {self.model_version}, Dimensions: {self.dimensions})")

    def _load_model_config(self) -> Dict[str, Any]:
        """Loads model-specific configuration (e.g., dimensions) from a JSON file."""
        config_path = os.path.join(
            os.path.dirname(__file__), 
            '..', 
            'models', 
            'embedding_models', 
            f"{self.model_id}_{self.model_version}.json"
        )
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Model config file not found: {config_path}. Using defaults.")
            return {"dimensions": 3072} # Default for text-embedding-3-large
        except json.JSONDecodeError:
            logger.error(f"Failed to decode model config file: {config_path}. Using defaults.")
            return {"dimensions": 3072}

    async def get_embedding(self, text: str, cache_key: str = None) -> Optional[List[float]]:
        """
        Gets the embedding for a single text string, using cache if available.
        
        Args:
            text (str): The text to embed.
            cache_key (str, optional): A unique key (e.g., event_id) for caching.
        
        Returns:
            Optional[List[float]]: The embedding vector, or None if an error occurred.
        """
        text = text.replace("\n", " ") # Per OpenAI recommendation

        if not text:
            logger.warning("Attempted to embed empty string.")
            return None

        # 1. Check cache
        s3_key = None
        if self.s3_client and self.s3_bucket and cache_key:
            s3_key = f"{self.cache_prefix}{cache_key}.json"
            cached_embedding = await self.s3_client.read_json(self.s3_bucket, s3_key)
            if cached_embedding:
                logger.debug(f"Retrieved embedding from cache: {s3_key}")
                return cached_embedding.get('embedding')

        # 2. Generate embedding if not in cache
        try:
            response = await self.client.embeddings.create(
                input=[text],
                model=self.model_id,
                dimensions=self.dimensions
            )
            embedding = response.data[0].embedding

            # 3. Store in cache
            if self.s3_client and self.s3_bucket and s3_key:
                cache_data = {
                    "cache_key": cache_key,
                    "model_id": self.model_id,
                    "model_version": self.model_version,
                    "text_snippet": text[:250], # For auditability
                    "embedding": embedding
                }
                await self.s3_client.write_json(self.s3_bucket, s3_key, cache_data)
                logger.debug(f"Cached new embedding to: {s3_key}")

            return embedding

        except Exception as e:
            logger.error(f"Error calling OpenAI Embedding API: {e}", exc_info=True)
            return None

    async def get_embeddings(self, texts: List[str], cache_keys: List[str] = None) -> List[Optional[List[float]]]:
        """
        Gets embeddings for a batch of texts.
        Note: This batch implementation currently calls get_embedding sequentially.
        For high throughput, this should be optimized to use asyncio.gather.
        """
        if cache_keys and len(texts) != len(cache_keys):
            logger.error("Mismatched length between texts and cache_keys.")
            raise ValueError("Texts and cache_keys must have the same length.")

        embeddings = []
        for i, text in enumerate(texts):
            key = cache_keys[i] if cache_keys else None
            embedding = await self.get_embedding(text, cache_key=key)
            embeddings.append(embedding)
            
        return embeddings

    def normalize_embedding(self, embedding: List[float]) -> List[float]:
        """
Examples:
        >>> client.normalize_embedding([1, 1, 1, 1])
        [0.5, 0.5, 0.5, 0.5]
        """
        if not embedding:
            return []
        
        norm = np.linalg.norm(embedding)
        if norm == 0:
            logger.warning("Attempted to normalize a zero-vector.")
            return embedding
        
        normalized = (np.array(embedding) / norm).tolist()
        return normalized

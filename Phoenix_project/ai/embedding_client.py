"""
Phoenix_project/ai/embedding_client.py
[Phase 2 Task 5] Fix Hardcoded Embedding Dimension.
Implement dynamic dimension detection via API probe or config override.
[P0-INFRA-04] Enhanced Reliability with Retry and Truncation.
"""
import logging
import asyncio
from typing import List, Optional, Union, Dict, Any
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from Phoenix_project.utils.retry import retry_with_exponential_backoff

from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

class EmbeddingClient:
    """
    Client for generating text embeddings using Google Gemini API.
    [Fix] Supports dynamic output dimension detection.
    """

    def __init__(
        self, 
        api_key: str, 
        model_name: str = "models/text-embedding-004", 
        provider: str = "google",
        logger: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the EmbeddingClient.

        Args:
            api_key: Gemini API Key.
            model_name: Model identifier.
            provider: 'google' or other supported providers.
            logger: Custom logger.
            config: Configuration dict (optional), can specify 'output_dimension'.
        """
        self.logger = logger or get_logger(__name__)
        self.api_key = api_key
        self.model_name = model_name
        self.provider = provider
        self.config = config or {}
        
        # [Fix] Configurable dimension override
        self._output_dimension = self.config.get("output_dimension")

        if not self.api_key:
            self.logger.warning("EmbeddingClient initialized without API Key. Calls will fail.")
        else:
            if self.provider == "google":
                genai.configure(api_key=self.api_key)
            # Add other providers setup here if needed

        # [Fix] Initialize dimension detection
        # We don't block __init__ with network calls, so we'll lazy load or async init if possible.
        # But since get_output_dimension is synchronous usually, we might need a stored value.
        # Let's rely on lazy detection on first access if not provided in config.
    
    def get_output_dimension(self) -> int:
        """
        Returns the embedding dimension size.
        [Fix] Dynamically detects dimension if not configured.
        """
        if self._output_dimension:
            return self._output_dimension
            
        # If not set, try to probe synchronously (fallback)
        # Note: Ideally this should be async, but often called in __init__ of VectorStore.
        # We perform a lightweight probe here.
        try:
            self.logger.info(f"Probing dimension for model {self.model_name}...")
            # Blocking call for safety during init chains
            # We create a new event loop for this sync call if we are not in one, 
            # OR we rely on a specialized sync method if the library supports it.
            # Google GenAI python lib has sync methods.
            
            if self.provider == "google":
                result = genai.embed_content(
                    model=self.model_name,
                    content="probe",
                    task_type="retrieval_document"
                )
                embedding = result['embedding']
                self._output_dimension = len(embedding)
                self.logger.info(f"Detected dimension: {self._output_dimension}")
                return self._output_dimension
            
            # Default fallback for unknown providers
            self.logger.warning("Unknown provider, defaulting dimension to 768.")
            return 768

        except Exception as e:
            self.logger.error(f"Failed to probe dimension: {e}")
            # Fallback to standard OpenAI/Gemini defaults to prevent crash
            if "large" in self.model_name: return 3072
            if "medium" in self.model_name: return 1536 # common
            return 768

    # [P0-INFRA-04] Safe batch wrapper with Retry and Truncation
    @retry_with_exponential_backoff(
        max_retries=3, 
        initial_backoff=1.0, 
        exceptions_to_retry=(google_exceptions.GoogleAPICallError, google_exceptions.RetryError, asyncio.TimeoutError)
    )
    async def _embed_batch_safe(self, batch: List[str]) -> List[List[float]]:
        """
        Helper to embed a batch with input truncation and retries.
        """
        # Truncate inputs to ~2000 tokens (approx 8000 chars) to avoid 400 errors
        truncated_batch = [text[:8000] for text in batch]
        
        def _call_api():
            return genai.embed_content(
                model=self.model_name,
                content=truncated_batch,
                task_type="retrieval_document"
            )
        
        # Offload sync API call to thread to prevent blocking loop
        result = await asyncio.to_thread(_call_api)
        
        if 'embedding' in result:
            return result['embedding']
        return []

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of texts asynchronously.
        """
        if not texts:
            return []

        if self.provider == "google":
            return await self._get_google_embeddings(texts)
        else:
            self.logger.error(f"Unsupported provider: {self.provider}")
            return []

    async def _get_google_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Internal handler for Google Gemini embeddings with batching and retry.
        """
        embeddings = []
        batch_size = 100 # Gemini batch limit
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            try:
                # [P0-INFRA-04] Use safe wrapper
                batch_embeddings = await self._embed_batch_safe(batch)
                
                # Verify result structure (single vs batch)
                if batch_embeddings and isinstance(batch_embeddings[0], (int, float)):
                     # Single vector returned for single input? Wrap it.
                     embeddings.append(batch_embeddings)
                else:
                     embeddings.extend(batch_embeddings)
                     
            except Exception as e:
                self.logger.error(f"Embedding batch {i//batch_size} failed: {e}")
                # Log error but continue to process valid batches (partial success)
                # Alternatively, append empty vectors to keep index alignment?
                # For RAG, skipping is usually safer than misalignment.
                pass

        return embeddings

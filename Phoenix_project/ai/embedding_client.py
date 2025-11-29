import asyncio
import time 
from typing import List, Optional, Any
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from google.api_core import retry_async 

from Phoenix_project.monitor.logging import get_logger
from Phoenix_project.core.exceptions import PhoenixError

class EmbeddingError(PhoenixError):
    """Exception raised for errors in the embedding client."""
    pass

logger = get_logger(__name__)

class EmbeddingClient:
    """
    负责从文本生成嵌入向量。
    """
    def __init__(self, provider: str, model_name: str, api_key: Optional[str] = None, batch_size: int = 32, logger: Optional[Any] = None):
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_key = api_key
        self.batch_size = batch_size 
        self.logger = logger or get_logger(__name__)

        self.model: Optional[SentenceTransformer] = None 
        self.google_client_configured = False

        self._load_model()
        self.logger.info(f"EmbeddingClient initialized with provider: {self.provider}, model: {self.model_name}")

    def _load_model(self):
        """
        加载 SentenceTransformer 模型或配置 Google 客户端。
        """
        if self.provider == 'google':
            if not self.api_key:
                self.logger.critical("Google provider selected but no API key provided.")
                raise EmbeddingError("Google API key (GEMINI_API_KEY) is missing.")
            try:
                genai.configure(api_key=self.api_key)
                self.google_client_configured = True
                self.logger.info("Google GenerativeAI client configured successfully.")
            except Exception as e:
                self.logger.critical(f"Failed to configure Google client: {e}")
                raise EmbeddingError(f"Could not configure Google client: {e}")

        else: 
            max_retries = 3
            delay_seconds = 2
            for attempt in range(max_retries):
                try:
                    self.logger.info(f"Loading local embedding model '{self.model_name}' (attempt {attempt + 1}/{max_retries})...")
                    self.model = SentenceTransformer(self.model_name)
                    self.logger.info(f"Embedding model '{self.model_name}' loaded successfully.")
                    return 
                except Exception as e:
                    self.logger.error(f"Failed to load embedding model '{self.model_name}': {e}")
                    if attempt < max_retries - 1:
                        self.logger.warning(f"Retrying in {delay_seconds} seconds...")
                        time.sleep(delay_seconds)
                    else:
                        self.logger.critical(f"Could not load embedding model after {max_retries} attempts.")
                        raise EmbeddingError(f"Could not load embedding model: {e}")
            
            if not self.model:
                raise EmbeddingError("Model loading failed after retries.")


    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        为一批文本异步生成嵌入向量。
        """
        if not self.google_client_configured and not self.model:
            self.logger.error("Model or client not loaded.")
            raise EmbeddingError("Model or client not loaded.")
            
        if not texts:
            return []
        
        self.logger.info(f"Embedding {len(texts)} texts using {self.provider} (task_type: RETRIEVAL_DOCUMENT)...")

        if self.google_client_configured:
            all_embeddings = []
            effective_batch_size = min(self.batch_size, 100) 

            for i in range(0, len(texts), effective_batch_size):
                batch = texts[i:i + effective_batch_size]
                try:
                    response = await genai.embed_content_async(
                        model=self.model_name,
                        content=batch,
                        task_type="RETRIEVAL_DOCUMENT"
                    )
                    all_embeddings.extend(response['embedding'])
                except Exception as e:
                    self.logger.error(f"Google embedding failed for batch: {e}")
                    raise EmbeddingError(f"Google embedding failed: {e}")
            return all_embeddings

        elif self.model:
            all_embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                try:
                    embeddings = await asyncio.to_thread(self._embed_batch, batch)
                    all_embeddings.extend(embeddings)
                except Exception as e:
                    self.logger.error(f"Failed to embed batch: {e}")
                    raise EmbeddingError(f"Failed during batch embedding: {e}")
            
            self.logger.info("Embedding complete.")
            return all_embeddings

        else:
            raise EmbeddingError("No valid embedding method available.")

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        处理单个批次的嵌入（同步）。
        """
        if not self.model:
             raise EmbeddingError("Local model not loaded for _embed_batch.")
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return [emb.tolist() for emb in embeddings]

    async def embed_query(self, query: str) -> List[float]:
        """
        为单个查询字符串生成嵌入向量。
        """
        if not self.google_client_configured and not self.model:
            self.logger.error("Model or client not loaded.")
            raise EmbeddingError("Model or client not loaded.")

        if self.google_client_configured:
            try:
                response = await genai.embed_content_async(
                    model=self.model_name,
                    content=query,
                    task_type="RETRIEVAL_QUERY"
                )
                return response['embedding']
            except Exception as e:
                self.logger.error(f"Failed to embed query with Google: {e}")
                raise EmbeddingError(f"Failed to embed query with Google: {e}")

        elif self.model:
            try:
                embedding = await asyncio.to_thread(self.model.encode, query)
                return embedding.tolist()
            except Exception as e:
                self.logger.error(f"Failed to embed query: {e}")
                raise EmbeddingError(f"Failed to embed query: {e}")

        else:
            raise EmbeddingError("No valid embedding method available for embed_query.")
            
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        get_embeddings (被 vector_store.py 调用)
        """
        return await self.embed(texts)

    def get_output_dimension(self) -> int:
        """
        [Phase III Fix] Returns the dimension of the embedding model.
        Supports dynamic switching between models (e.g. 768 vs 3072).
        """
        if "text-embedding-3-large" in self.model_name:
            return 3072
        elif "text-embedding-3-small" in self.model_name:
            return 1536
        elif "text-embedding-004" in self.model_name:
            return 768
        # Fallback for known local models
        elif "all-MiniLM-L6-v2" in self.model_name:
            return 384
        else:
            # Default fallback
            return 768

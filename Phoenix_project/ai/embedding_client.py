import asyncio
import time # 修复：导入 time
from typing import List, Optional, Any
from sentence_transformers import SentenceTransformer # type: ignore
# --- Google Imports 喵! ---
import google.generativeai as genai
from google.api_core import retry_async # (可选，用于异步重试)

# 修复：导入正确的 monitor.logging 和 core.exceptions
from Phoenix_project.monitor.logging import get_logger
from Phoenix_project.core.exceptions import PhoenixError

# 修复：定义 EmbeddingError
class EmbeddingError(PhoenixError):
    """Exception raised for errors in the embedding client."""
    pass

# 修复：使用 get_logger
logger = get_logger(__name__)

class EmbeddingClient:
    """
    负责从文本生成嵌入向量。
    """
    def __init__(self, provider: str, model_name: str, api_key: Optional[str] = None, batch_size: int = 32, logger: Optional[Any] = None):
        self.provider = provider.lower()
        self.model_name = model_name
        self.api_key = api_key
        self.batch_size = batch_size # (Google API 批处理限制是 100)
        # 修复：使用传入的 logger
        self.logger = logger or get_logger(__name__)

        self.model: Optional[SentenceTransformer] = None # (本地 SentenceTransformer)
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
                # Google SDK 是全局配置的
                genai.configure(api_key=self.api_key)
                self.google_client_configured = True
                self.logger.info("Google GenerativeAI client configured successfully.")
            except Exception as e:
                self.logger.critical(f"Failed to configure Google client: {e}")
                raise EmbeddingError(f"Could not configure Google client: {e}")

        else: # 假设 'local' 或其他 (默认 SentenceTransformer)
            max_retries = 3
            delay_seconds = 2
            for attempt in range(max_retries):
                try:
                    self.logger.info(f"Loading local embedding model '{self.model_name}' (attempt {attempt + 1}/{max_retries})...")
                    self.model = SentenceTransformer(self.model_name)
                    self.logger.info(f"Embedding model '{self.model_name}' loaded successfully.")
                    return # 成功
                except Exception as e:
                    self.logger.error(f"Failed to load embedding model '{self.model_name}': {e}")
                    if attempt < max_retries - 1:
                        self.logger.warning(f"Retrying in {delay_seconds} seconds...")
                        time.sleep(delay_seconds)
                    else:
                        self.logger.critical(f"Could not load embedding model after {max_retries} attempts.")
                        raise EmbeddingError(f"Could not load embedding model: {e}")
            
            # 这行理论上不应该被执行到
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
            # --- Google API 逻辑 ---
            all_embeddings = []
            # (Google API 限制 100 个/批次)
            effective_batch_size = min(self.batch_size, 100) 

            for i in range(0, len(texts), effective_batch_size):
                batch = texts[i:i + effective_batch_size]
                try:
                    # 异步运行
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
            # --- 本地 SentenceTransformer 逻辑 (保持不变) ---
            all_embeddings = []
            # 创建批次
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                try:
                    # 在单独的线程中运行同步的、受CPU限制的批处理嵌入
                    embeddings = await asyncio.to_thread(self._embed_batch, batch)
                    all_embeddings.extend(embeddings)
                except Exception as e:
                    self.logger.error(f"Failed to embed batch: {e}")
                    # 根据策略决定：是继续还是引发异常
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
        # encode 方法是同步的且受CPU限制
        embeddings = self.model.encode(texts, show_progress_bar=False)
        # 转换为标准的 Python 列表
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
                # 在单独的线程中运行
                embedding = await asyncio.to_thread(self.model.encode, query)
                return embedding.tolist()
            except Exception as e:
                self.logger.error(f"Failed to embed query: {e}")
                raise EmbeddingError(f"Failed to embed query: {e}")

        else:
            raise EmbeddingError("No valid embedding method available for embed_query.")
            
    # 修复：添加 get_embeddings (被 vector_store.py 调用)
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        get_embeddings (被 vector_store.py 调用)
        """
        return await self.embed(texts)

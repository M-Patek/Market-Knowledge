import asyncio
import time # 修复：导入 time
from typing import List, Optional
from sentence_transformers import SentenceTransformer # type: ignore
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
    def __init__(self, model_name: str, batch_size: int = 32, logger: Optional[Any] = None):
        self.model_name = model_name
        self.batch_size = batch_size
        # 修复：使用传入的 logger
        self.logger = logger or get_logger(__name__)
        self.model = self._load_model()
        self.logger.info(f"EmbeddingClient initialized with model: {self.model_name}")

    def _load_model(self) -> SentenceTransformer:
        """
        加载 SentenceTransformer 模型。
        修复：添加重试逻辑。
        """
        max_retries = 3
        delay_seconds = 2
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Loading embedding model '{self.model_name}' (attempt {attempt + 1}/{max_retries})...")
                model = SentenceTransformer(self.model_name)
                self.logger.info(f"Embedding model '{self.model_name}' loaded successfully.")
                return model
            except Exception as e:
                self.logger.error(f"Failed to load embedding model '{self.model_name}': {e}")
                if attempt < max_retries - 1:
                    self.logger.warning(f"Retrying in {delay_seconds} seconds...")
                    time.sleep(delay_seconds)
                else:
                    self.logger.critical(f"Could not load embedding model after {max_retries} attempts.")
                    raise EmbeddingError(f"Could not load embedding model: {e}")
        
        # 这行理论上不应该被执行到
        raise EmbeddingError("Model loading failed after retries.")


    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        为一批文本异步生成嵌入向量。
        """
        if not self.model:
            self.logger.error("Embedding model is not loaded.")
            raise EmbeddingError("Model not loaded.")
            
        if not texts:
            return []

        self.logger.info(f"Embedding {len(texts)} texts in batches of {self.batch_size}...")
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

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        处理单个批次的嵌入（同步）。
        """
        # encode 方法是同步的且受CPU限制
        embeddings = self.model.encode(texts, show_progress_bar=False)
        # 转换为标准的 Python 列表
        return [emb.tolist() for emb in embeddings]

    async def embed_query(self, query: str) -> List[float]:
        """
        为单个查询字符串生成嵌入向量。
        """
        if not self.model:
            self.logger.error("Embedding model is not loaded.")
            raise EmbeddingError("Model not loaded.")
            
        try:
            # 在单独的线程中运行
            embedding = await asyncio.to_thread(self.model.encode, query)
            return embedding.tolist()
        except Exception as e:
            self.logger.error(f"Failed to embed query: {e}")
            raise EmbeddingError(f"Failed to embed query: {e}")
            
    # 修复：添加 get_embeddings (被 vector_store.py 调用)
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        get_embeddings (被 vector_store.py 调用)
        """
        return await self.embed(texts)

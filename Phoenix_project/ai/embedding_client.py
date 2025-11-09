import asyncio
from typing import List, Optional
from sentence_transformers import SentenceTransformer # type: ignore
from monitor.logging import logger
from core.exceptions import EmbeddingError

class EmbeddingClient:
    """
    负责从文本生成嵌入向量。
    """
    def __init__(self, model_name: str, batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = self._load_model()
        logger.info(f"EmbeddingClient initialized with model: {self.model_name}")

    def _load_model(self) -> SentenceTransformer:
        """
        加载 SentenceTransformer 模型。
        """
        try:
            # TODO: Add retry logic
            return SentenceTransformer(self.model_name)
        except Exception as e:
            logger.error(f"Failed to load embedding model '{self.model_name}': {e}")
            raise EmbeddingError(f"Could not load embedding model: {e}")

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """
        为一批文本异步生成嵌入向量。
        """
        if not self.model:
            logger.error("Embedding model is not loaded.")
            raise EmbeddingError("Model not loaded.")
            
        if not texts:
            return []

        logger.info(f"Embedding {len(texts)} texts in batches of {self.batch_size}...")
        all_embeddings = []
        
        # 创建批次
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            try:
                # 在单独的线程中运行同步的、受CPU限制的批处理嵌入
                embeddings = await asyncio.to_thread(self._embed_batch, batch)
                all_embeddings.extend(embeddings)
            except Exception as e:
                logger.error(f"Failed to embed batch: {e}")
                # 根据策略决定：是继续还是引发异常
                raise EmbeddingError(f"Failed during batch embedding: {e}")

        logger.info("Embedding complete.")
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
            logger.error("Embedding model is not loaded.")
            raise EmbeddingError("Model not loaded.")
            
        try:
            # 在单独的线程中运行
            embedding = await asyncio.to_thread(self.model.encode, query)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to embed query: {e}")
            raise EmbeddingError(f"Failed to embed query: {e}")

import os
import uuid
import asyncio
import datetime 
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from pinecone import Pinecone, ServerlessSpec
from uuid import UUID

from Phoenix_project.monitor.logging import ESLogger, get_logger
from Phoenix_project.ai.embedding_client import EmbeddingClient
from Phoenix_project.utils.retry import retry_with_exponential_backoff

class Document:
    def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
        self.page_content = page_content
        self.metadata = metadata or {}

    def __repr__(self):
        return f"Document(page_content='{self.page_content[:50]}...', metadata={self.metadata})"


class BaseVectorStore(ABC):
    """Abstract interface for a vector store."""

    @abstractmethod
    async def aadd_batch(self, batch: List[Document], embeddings: List[List[float]], batch_id: UUID):
        pass

    @abstractmethod
    async def count_by_batch_id(self, batch_id: UUID) -> int:
        pass

    @abstractmethod
    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        pass

class MockVectorStore(BaseVectorStore):
    """
    [已保留] 一个简单的内存模拟向量存储。
    """
    def __init__(self, embedding_client: EmbeddingClient, logger: ESLogger, config: Dict[str, Any]):
        self.embedding_client = embedding_client
        self.logger = logger or get_logger(__name__) 
        self.config = config
        self.store: Dict[str, Dict[str, Any]] = {}  
        self.lock = asyncio.Lock()
        self.logger.info("MockVectorStore initialized (in-memory fallback).")
        
        try:
            import numpy as np
            self.np = np
        except ImportError:
            self.logger.error("Numpy not found. MockVectorStore search will not work.")
            self.np = None

    async def aadd_batch(self, batch: List[Document], embeddings: List[List[float]], batch_id: UUID):
        ids = []
        try:
            async with self.lock:
                for doc, vector in zip(batch, embeddings):
                    doc_id = doc.metadata.get("id", f"doc_{hash(doc.page_content)}")
                    doc.metadata["ingestion_batch_id"] = str(batch_id)
                    self.store[doc_id] = {"vector": vector, "document": doc}
                    ids.append(doc_id)
            self.logger.info(f"MockVectorStore: Added {len(batch)} documents to batch {batch_id}.")
            return ids
        except Exception as e:
            self.logger.error(f"Failed to add documents to MockVectorStore: {e}", exc_info=True)
            return []

    async def count_by_batch_id(self, batch_id: UUID) -> int:
        count = 0
        for data in self.store.values():
            if data["document"].metadata.get("ingestion_batch_id") == str(batch_id):
                count += 1
        return count

    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        if self.np is None: return []
        if not self.store: return []

        try:
            query_embedding = await self.embedding_client.get_embeddings([query])
            if not query_embedding: return []
            
            q_vec = self.np.array(query_embedding[0])
            q_norm = self.np.linalg.norm(q_vec)
            if q_norm == 0: return []
            
            scores = []
            async with self.lock:
                for doc_id, data in self.store.items():
                    doc_vec = self.np.array(data["vector"])
                    doc_norm = self.np.linalg.norm(doc_vec)
                    if doc_norm == 0: continue
                    sim = self.np.dot(q_vec, doc_vec) / (q_norm * doc_norm)
                    scores.append((data["document"], float(sim)))
            
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:k]

        except Exception as e:
            self.logger.error(f"Failed during similarity search: {e}", exc_info=True)
            return []


class PineconeVectorStore(BaseVectorStore):
    """
    [已优化] Pinecone Serverless 向量存储。
    [Phase III Fix] Dynamic Dimension Sync & Unified Namespace.
    """
    def __init__(
        self, embedding_client: EmbeddingClient, logger: ESLogger, config: Dict[str, Any]
    ):
        self.embedding_client = embedding_client
        self.logger = logger or get_logger(__name__)
        self.config = config
        
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            self.logger.error("PINECONE_API_KEY 未在环境变量中设置！")
            raise ValueError("PINECONE_API_KEY not found in environment.")
            
        try:
            self.pc = Pinecone(api_key=api_key)
            self.index_name = self.config.get("index_name", "phoenix-project-rag")
            self.global_namespace = self.config.get("global_namespace", "phoenix_knowledge_v1")
            
            # [Phase III Fix] Dynamic Dimension Sync
            self.dimension = self.embedding_client.get_output_dimension()
            
            # 检查索引是否存在
            if self.index_name not in self.pc.list_indexes().names():
                self.logger.warning(f"Pinecone 索引 '{self.index_name}' 未找到。正在尝试创建...")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine", 
                    spec=ServerlessSpec(cloud="aws", region="us-west-2")
                )
                self.logger.info(f"Pinecone 索引 '{self.index_name}' 已创建 (Dim: {self.dimension})。")
            
            self.index = self.pc.Index(self.index_name)
            self.logger.info(f"PineconeVectorStore 已连接到索引 '{self.index_name}'.")
            
        except Exception as e:
            self.logger.error(f"初始化 PineconeVectorStore 失败: {e}", exc_info=True)
            raise

    @retry_with_exponential_backoff(max_retries=3, initial_backoff=1.0)
    async def _execute_upsert(self, vectors, namespace):
        """[Task 2] Async wrapper for Pinecone upsert with retry logic."""
        return await asyncio.to_thread(
            self.index.upsert,
            vectors=vectors,
            namespace=namespace
        )

    @retry_with_exponential_backoff(max_retries=3, initial_backoff=1.0)
    async def _execute_query(self, **kwargs):
        """[Task 2] Async wrapper for Pinecone query with retry logic."""
        return await asyncio.to_thread(
            self.index.query,
            **kwargs
        )

    async def aadd_batch(self, batch: List[Document], embeddings: List[List[float]], batch_id: UUID):
        """
        [Task 3C] Add batch to Global Namespace.
        [Phase I Fix] Optimized Metadata (Reduced payload).
        """
        if not self.index:
            raise Exception("Pinecone index not initialized.")
        
        vectors_to_upsert = []
        for doc, vector in zip(batch, embeddings):
            doc_id = doc.metadata.get("doc_id", doc.id if hasattr(doc, 'id') else str(uuid.uuid4()))
            
            # [Phase I Fix] Sanitize and minimize metadata
            sanitized_metadata = {
                "doc_id": doc_id,
                "source": doc.metadata.get("source", "unknown"),
                "content_preview": doc.page_content[:100] if doc.page_content else "",
                "ingestion_batch_id": str(batch_id)
            }
            
            vectors_to_upsert.append({
                "id": doc_id,
                "values": vector,
                "metadata": sanitized_metadata
            })
        
        if vectors_to_upsert:
            self.logger.info(f"Pinecone: Upserting {len(vectors_to_upsert)} vectors to namespace '{self.global_namespace}'...")
            await self._execute_upsert(vectors=vectors_to_upsert, namespace=self.global_namespace)
            return [v['id'] for v in vectors_to_upsert]
        return []

    async def count_by_batch_id(self, batch_id: UUID) -> int:
        """
        [Task 3C] Count by metadata filtering (since we no longer isolate namespaces).
        Note: Pinecone describe_index_stats doesn't support metadata filtering efficiently.
        This is an approximation or requires a dummy query.
        """
        if not self.index:
            return 0
        # Return 0 or implement a dummy query if critical.
        # For performance, we skip exact count in global namespace mode unless needed.
        return 0

    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]: 
        """[已优化] 在 Pinecone 中执行异步相似性搜索。"""
        
        pinecone_filter = kwargs.get("filter") 
        # [Phase I Fix] Default to Global Namespace
        namespace_to_search = kwargs.get("namespace", self.global_namespace)

        try:
            query_embedding = await self.embedding_client.get_embeddings([query])
            if not query_embedding:
                self.logger.error("无法获取查询的嵌入。")
                return []
            
            self.logger.debug(f"正在 Pinecone 中查询 (Filter: {pinecone_filter}, NS: {namespace_to_search})")
            
            results = await self._execute_query(
                vector=query_embedding[0],
                top_k=k,
                include_metadata=True,
                filter=pinecone_filter,
                namespace=namespace_to_search
            )
            
            docs_with_scores = []
            if results and "matches" in results:
                for match in results["matches"]:
                    metadata = match.get("metadata", {})
                    # Reconstruct Document from preview
                    page_content = metadata.get("content_preview", "")
                    score = float(match.get("score", 0.0))
                    
                    doc = Document(page_content=page_content, metadata=metadata)
                    docs_with_scores.append((doc, score))
            
            return docs_with_scores

        except Exception as e:
            self.logger.error(f"Pinecone 相似性搜索失败: {e}", exc_info=True)
            return []

    async def adelete_namespace(self, namespace: str) -> bool:
        if not namespace: return False
        try:
            await asyncio.to_thread(self.index.delete, namespace=namespace, delete_all=True)
            return True
        except Exception as e:
            self.logger.error(f"删除命名空间 {namespace} 失败: {e}", exc_info=True)
            return False

def get_vector_store(config: Dict[str, Any], embedding_client: EmbeddingClient, logger: ESLogger) -> BaseVectorStore:
    if logger is None: logger = get_logger(__name__)
    store_type = config.get("type", "pinecone")
    store_config = config

    if store_type == "pinecone":
        if os.environ.get("PINECONE_API_KEY"):
            try:
                return PineconeVectorStore(embedding_client, logger, store_config)
            except Exception as e:
                logger.error(f"初始化 PineconeVectorStore 失败: {e}。回退到 Mock。")
                return MockVectorStore(embedding_client, logger, store_config)
        else:
            return MockVectorStore(embedding_client, logger, store_config)
    else:
        return MockVectorStore(embedding_client, logger, store_config)

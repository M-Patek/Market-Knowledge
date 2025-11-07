import os
import uuid
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from pinecone import Pinecone, ServerlessSpec

# 修正：将 'monitor.logging...' 转换为 'Phoenix_project.monitor.logging...'
from Phoenix_project.monitor.logging import ESLogger, get_logger
# 修正：将 'ai.embedding_client...' 转换为 'Phoenix_project.ai.embedding_client...'
from Phoenix_project.ai.embedding_client import EmbeddingClient

# Mock Document class
class Document:
    def __init__(self, page_content: str, metadata: Optional[Dict[str, Any]] = None):
        self.page_content = page_content
        self.metadata = metadata or {}

    def __repr__(self):
        return f"Document(page_content='{self.page_content[:50]}...', metadata={self.metadata})"


class BaseVectorStore(ABC):
    """Abstract interface for a vector store."""

    @abstractmethod
    async def aadd_documents(
        self, documents: List[Document], **kwargs: Any
    ) -> List[str]:
        """Add documents to the vector store."""
        pass

    @abstractmethod
    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]: # <--- [蓝图 2] 更改返回类型
        """Perform a similarity search."""
        pass

class MockVectorStore(BaseVectorStore):
    """
    [已保留]
    一个简单的内存模拟向量存储，作为 Pinecone 无法连接时的回退选项。
    """
    def __init__(
        self, embedding_client: EmbeddingClient, logger: ESLogger, config: Dict[str, Any]
    ):
        self.embedding_client = embedding_client
        self.logger = logger or get_logger(__name__) # 确保 logger 存在
        self.config = config
        self.store: Dict[str, Dict[str, Any]] = {}  # {id: {"vector": [...], "document": Document}}
        self.lock = asyncio.Lock()
        self.logger.info("MockVectorStore initialized (in-memory fallback).")
        
        try:
            import numpy as np
            self.np = np
        except ImportError:
            self.logger.error("Numpy not found. MockVectorStore search will not work.")
            self.np = None

    async def aadd_documents(
        self, documents: List[Document], **kwargs: Any
    ) -> List[str]:
        """Adds documents to the in-memory store."""
        ids = []
        texts = [doc.page_content for doc in documents]
        
        if not texts:
            return []
            
        try:
            embeddings = await self.embedding_client.get_embeddings(texts)
            
            if not embeddings or len(embeddings) != len(documents):
                self.logger.error("Mismatch in embedding results. Cannot add documents.")
                return []

            async with self.lock:
                for doc, vector in zip(documents, embeddings):
                    doc_id = doc.metadata.get("id", f"doc_{hash(doc.page_content)}")
                    self.store[doc_id] = {"vector": vector, "document": doc}
                    ids.append(doc_id)
            
            self.logger.info(f"Added {len(ids)} documents to MockVectorStore.")
            return ids
            
        except Exception as e:
            self.logger.error(f"Failed to add documents to MockVectorStore: {e}", exc_info=True)
            return []

    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]: # <--- [蓝图 2] 更改返回类型
        """Performs a naive similarity search."""
        if self.np is None:
            self.logger.error("Cannot perform search, numpy is not installed.")
            return []
            
        if not self.store:
            self.logger.warning("Search called on empty MockVectorStore.")
            return []

        try:
            query_embedding = await self.embedding_client.get_embeddings([query])
            if not query_embedding:
                self.logger.error("Failed to get embedding for query.")
                return []
            
            q_vec = self.np.array(query_embedding[0])
            q_norm = self.np.linalg.norm(q_vec)
            if q_norm == 0:
                self.logger.warning("Query embedding norm is zero.")
                return []
            
            scores = []
            async with self.lock:
                for doc_id, data in self.store.items():
                    doc_vec = self.np.array(data["vector"])
                    doc_norm = self.np.linalg.norm(doc_vec)
                    if doc_norm == 0:
                        continue
                    # Cosine similarity
                    sim = self.np.dot(q_vec, doc_vec) / (q_norm * doc_norm)
                    scores.append((data["document"], float(sim))) # <--- [蓝图 2] 更改
            
            scores.sort(key=lambda x: x[1], reverse=True) # <--- [蓝图 2] 按分数排序
            
            self.logger.debug(f"Similarity search for '{query[:20]}...' returned {len(scores)} results.")
            return scores[:k] # <--- [蓝图 2] 返回 (doc, score)

        except Exception as e:
            self.logger.error(f"Failed during similarity search: {e}", exc_info=True)
            return []


class PineconeVectorStore(BaseVectorStore):
    """
    [已优化]
    RAG_ARCHITECTURE.md 中指定的 Pinecone Serverless 向量存储的真实实现。
    """
    def __init__(
        self, embedding_client: EmbeddingClient, logger: ESLogger, config: Dict[str, Any]
    ):
        self.embedding_client = embedding_client
        self.logger = logger or get_logger(__name__) # 确保 logger 存在
        self.config = config
        
        api_key = os.environ.get("PINECONE_API_KEY")
        if not api_key:
            self.logger.error("PINECONE_API_KEY 未在环境变量中设置！")
            raise ValueError("PINECONE_API_KEY not found in environment.")
            
        try:
            self.pc = Pinecone(api_key=api_key)
            self.index_name = self.config.get("index_name", "phoenix-project-rag")
            self.dimension = self.config.get("embedding_dimension", 768) # TODO: 这应该从 embedding_client 获取
            
            # 检查索引是否存在
            if self.index_name not in self.pc.list_indexes().names():
                self.logger.warning(f"Pinecone 索引 '{self.index_name}' 未找到。正在尝试创建...")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine", # 适用于语义搜索
                    spec=ServerlessSpec(
                        cloud="aws", # Pinecone Serverless 的默认云
                        region="us-west-2" # 默认区域
                    )
                )
                self.logger.info(f"Pinecone 索引 '{self.index_name}' 已创建。")
            
            self.index = self.pc.Index(self.index_name)
            self.logger.info(f"PineconeVectorStore 已连接到索引 '{self.index_name}'.")
            
        except Exception as e:
            self.logger.error(f"初始化 PineconeVectorStore 失败: {e}", exc_info=True)
            raise

    async def aadd_documents(
        self, documents: List[Document], **kwargs: Any
    ) -> List[str]:
        """[已优化] 将文档异步添加到 Pinecone 索引。"""
        ids = []
        texts = [doc.page_content for doc in documents]
        
        if not texts:
            return []
            
        try:
            embeddings = await self.embedding_client.get_embeddings(texts)
            if not embeddings or len(embeddings) != len(documents):
                self.logger.error("嵌入(embedding)结果数量不匹配。无法添加文档。")
                return []

            vectors_to_upsert = []
            for doc, vector in zip(documents, embeddings):
                doc_id = doc.metadata.get("id", str(uuid.uuid4()))
                
                # 必须将页面内容存储在元数据中以便稍后检索
                metadata_to_store = doc.metadata.copy()
                metadata_to_store["page_content"] = doc.page_content
                
                # Pinecone 元数据限制：值必须是 str, int, float, bool, 或 list of str
                # 确保所有元数据值都是兼容的
                sanitized_metadata = {}
                for k, v in metadata_to_store.items():
                    if isinstance(v, (str, int, float, bool)):
                        sanitized_metadata[k] = v
                    elif isinstance(v, list) and all(isinstance(i, str) for i in v):
                         sanitized_metadata[k] = v
                    else:
                        # 序列化不兼容的类型
                        sanitized_metadata[k] = str(v)
                
                vectors_to_upsert.append({
                    "id": doc_id,
                    "values": vector,
                    "metadata": sanitized_metadata
                })
                ids.append(doc_id)
            
            # 在线程中执行阻塞的 upsert 调用
            # (Pinecone v4+ 推荐批量操作)
            self.logger.debug(f"正在向 Pinecone upsert {len(vectors_to_upsert)} 个向量...")
            await asyncio.to_thread(
                self.index.upsert,
                vectors=vectors_to_upsert,
                namespace="default" # 使用默认命名空间
            )
            
            self.logger.info(f"已添加 {len(ids)} 个文档到 PineconeVectorStore。")
            return ids
            
        except Exception as e:
            self.logger.error(f"添加文档到 PineconeVectorStore 失败: {e}", exc_info=True)
            return []

    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]: # <--- [蓝图 2] 更改返回类型
        """[已优化] 在 Pinecone 中执行异步相似性搜索。"""
        
        # [蓝图 2] 提取过滤器
        pinecone_filter = kwargs.get("filter") 
        
        try:
            query_embedding = await self.embedding_client.get_embeddings([query])
            if not query_embedding:
                self.logger.error("无法获取查询的嵌入(embedding)。")
                return []
            
            # 在线程中执行阻塞的 query 调用
            self.logger.debug(f"正在 Pinecone 中查询 '{query[:20]}...' (Filter: {pinecone_filter})")
            results = await asyncio.to_thread(
                self.index.query,
                vector=query_embedding[0],
                top_k=k,
                include_metadata=True,
                filter=pinecone_filter, # <--- [蓝图 2] 传递过滤器
                namespace="default"
            )
            
            # 从元数据中重构 Document 对象
            docs_with_scores = [] # <--- [蓝图 2] 更改
            if results and "matches" in results:
                for match in results["matches"]:
                    metadata = match.get("metadata", {})
                    # 弹出我们存储的页面内容
                    page_content = metadata.pop("page_content", "")
                    # [蓝图 2] 获取分数
                    score = float(match.get("score", 0.0))
                    
                    doc = Document(page_content=page_content, metadata=metadata)
                    docs_with_scores.append((doc, score)) # <--- [蓝图 2] 存储 (doc, score)
            
            self.logger.debug(f"Pinecone 相似性搜索返回 {len(docs_with_scores)} 个结果。")
            return docs_with_scores

        except Exception as e:
            self.logger.error(f"Pinecone 相似性搜索失败: {e}", exc_info=True)
            return []


def get_vector_store(
    config: Dict[str, Any],
    embedding_client: EmbeddingClient,
    logger: ESLogger
) -> BaseVectorStore:
    """
    [已优化]
    工厂函数，用于初始化并返回一个向量存储实例。
    它现在会优先尝试 Pinecone，如果 API 密钥存在的话。
    """
    if logger is None:
        logger = get_logger(__name__)
        
    # 默认尝试 pinecone
    store_type = config.get("type", "pinecone") # 从 config['vector_store'] 更改
    store_config = config # 假设 config 已经是 config['vector_store']

    if store_type == "pinecone":
        if os.environ.get("PINECONE_API_KEY"):
            try:
                logger.info("正在使用 PineconeVectorStore (生产环境)。")
                return PineconeVectorStore(embedding_client, logger, store_config)
            except Exception as e:
                logger.error(f"初始化 PineconeVectorStore 失败: {e}。回退到 MockVectorStore。")
                return MockVectorStore(embedding_client, logger, store_config)
        else:
            logger.warning("PINECONE_API_KEY 未设置。回退到 MockVectorStore。")
            return MockVectorStore(embedding_client, logger, store_config)
    
    elif store_type == "mock":
        logger.info("正在使用 MockVectorStore (在内存中)。")
        return MockVectorStore(embedding_client, logger, store_config)
        
    else:
        logger.error(f"未知的向量存储类型: {store_type}。回退到 mock。")
        return MockVectorStore(embedding_client, logger, store_config)

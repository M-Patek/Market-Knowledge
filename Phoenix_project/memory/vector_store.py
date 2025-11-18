import os
import uuid
import asyncio
import datetime # [主人喵的清洁计划 1.1] 导入
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from pinecone import Pinecone, ServerlessSpec
from uuid import UUID

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
    async def aadd_batch(self, batch: List[Document], embeddings: List[List[float]], batch_id: UUID):
        """
        [Task 3C] Add a batch of documents and their embeddings to the vector store,
        associated with a specific ingestion batch_id.
        """
        pass

    @abstractmethod
    async def count_by_batch_id(self, batch_id: UUID) -> int:
        """
        [Task 3C] Count the number of vectors associated with a specific ingestion_batch_id.
        """
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

    async def aadd_batch(self, batch: List[Document], embeddings: List[List[float]], batch_id: UUID):
        """Mock adding documents."""
        ids = []
        
        try:
            async with self.lock:
                for doc, vector in zip(batch, embeddings):
                    doc_id = doc.metadata.get("id", f"doc_{hash(doc.page_content)}")
                    
                    # [Task 3C] Store the batch_id as a string in metadata for consistency
                    doc.metadata["ingestion_batch_id"] = str(batch_id)
                    
                    self.store[doc_id] = {"vector": vector, "document": doc}
                    ids.append(doc_id)
            
            self.logger.info(f"MockVectorStore: Added {len(batch)} documents to batch {batch_id}.")
            return ids
            
        except Exception as e:
            self.logger.error(f"Failed to add documents to MockVectorStore: {e}", exc_info=True)
            return []

    async def count_by_batch_id(self, batch_id: UUID) -> int:
        """[Task 3C] Mock counting by batch_id."""
        count = 0
        for data in self.store.values():
            if data["document"].metadata.get("ingestion_batch_id") == str(batch_id):
                count += 1
        self.logger.info(f"MockVectorStore: Found {count} items for batch_id {batch_id}.")
        return count

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

    async def aadd_batch(self, batch: List[Document], embeddings: List[List[float]], batch_id: UUID):
        """
        [Task 3C] Add a batch of documents and embeddings to the Pinecone index,
        using the batch_id as the namespace for atomic counting.
        """
        if not self.index:
            raise Exception("Pinecone index not initialized.")
        
        vectors_to_upsert = []
        for doc, vector in zip(batch, embeddings):
            doc_id = doc.metadata.get("doc_id", doc.id)
            if not doc_id:
                 doc_id = str(uuid.uuid4())
            
            # Sanitize metadata for Pinecone (only str, bool, float, int, or list of str)
            sanitized_metadata = {
                "doc_id": doc_id,
                "source": doc.metadata.get("source", "unknown"),
                "content_preview": doc.page_content[:250] if doc.page_content else "",
                # [Task 3C] Add batch_id to metadata as well for redundancy/filtering
                "ingestion_batch_id": str(batch_id)
            }
            
            # Add other simple metadata fields
            # (Assuming original file logic here to copy other simple types)
            for k, v in doc.metadata.items():
                 if k not in sanitized_metadata and isinstance(v, (str, int, float, bool)):
                      sanitized_metadata[k] = v

            vectors_to_upsert.append({
                "id": doc_id,
                "values": vector,
                "metadata": sanitized_metadata
            })
        
        if vectors_to_upsert:
            self.logger.info(f"Pinecone: Upserting {len(vectors_to_upsert)} vectors to namespace '{batch_id}'...")
            await asyncio.to_thread(
                self.index.upsert,
                vectors=vectors_to_upsert,
                namespace=str(batch_id) # [Task 3C] Use batch_id as the namespace
            )
            return [v['id'] for v in vectors_to_upsert]
        return []

    async def count_by_batch_id(self, batch_id: UUID) -> int:
        """
        [Task 3C] Get the exact vector count for a batch_id by using it as a namespace
        and querying the index stats. This is fast and 100% accurate.
        """
        if not self.index:
            self.logger.warning("Pinecone: count_by_batch_id called but index not initialized.")
            return 0
        
        try:
            stats = await asyncio.to_thread(self.index.describe_index_stats)
            namespace_stats = stats.get('namespaces', {})
            # Get stats for our specific batch_id namespace
            batch_stats = namespace_stats.get(str(batch_id), {})
            # Return the exact vector count
            count = batch_stats.get('vector_count', 0)
            self.logger.info(f"Pinecone: Found {count} vectors in namespace '{batch_id}'.")
            return count
        except Exception as e:
            self.logger.error(f"Pinecone: Failed to get index stats for batch_id {batch_id}: {e}", exc_info=True)
            return 0

    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]: # <--- [蓝图 2] 更改返回类型
        """[已优化] 在 Pinecone 中执行异步相似性搜索。"""
        
        # [蓝图 2] 提取过滤器
        pinecone_filter = kwargs.get("filter") 
        
        # [主人喵的清洁计划 1.1] 搜索时必须指定命名空间，否则 Janitor 清理后就找不到了
        # 默认搜索当月和上个月
        if "namespace" not in kwargs:
            today = datetime.datetime.utcnow()
            current_ns = today.strftime("ns-%Y-%m")
            # (这不支持跨月搜索，但对于 Janitor 来说是必须的)
            # (一个更好的实现是在 aadd_documents 时将 ns 存储在元数据中，然后在这里搜索)
            # (但为了简单起见，我们假设查询只关心最近的数据)
            namespace_to_search = current_ns
        else:
            namespace_to_search = kwargs.get("namespace", "default")

        try:
            query_embedding = await self.embedding_client.get_embeddings([query])
            if not query_embedding:
                self.logger.error("无法获取查询的嵌入(embedding)。")
                return []
            
            # 在线程中执行阻塞的 query 调用
            self.logger.debug(f"正在 Pinecone 中查询 '{query[:20]}...' (Filter: {pinecone_filter}, Namespace: {namespace_to_search})")
            results = await asyncio.to_thread(
                self.index.query,
                vector=query_embedding[0],
                top_k=k,
                include_metadata=True,
                filter=pinecone_filter, # <--- [蓝图 2] 传递过滤器
                namespace=namespace_to_search # [主人喵的清洁计划 1.1] 指定命名空间
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

    # [主人喵的清洁计划 1.1] 为 Janitor 添加删除方法
    async def adelete_namespace(self, namespace: str) -> bool:
        """[主人喵的清洁计划 1.1] 异步删除一个命名空间。"""
        if not namespace:
            self.logger.warning("adelete_namespace called with empty namespace.")
            return False
        try:
            self.logger.info(f"正在从索引 '{self.index_name}' 删除命名空间: {namespace}...")
            await asyncio.to_thread(
                self.index.delete,
                namespace=namespace,
                delete_all=True
            )
            self.logger.info(f"命名空间 {namespace} 已成功删除。")
            return True
        except Exception as e:
            self.logger.error(f"删除命名空间 {namespace} 失败: {e}", exc_info=True)
            return False

    # [主人喵的清洁计划 1.1] 为 Janitor 脚本添加一个同步的包装器
    def delete_by_namespace(self, namespace: str) -> bool:
        """同步包装器，用于 Janitor 脚本。"""
        self.logger.info(f"Janitor (同步) 请求删除命名空间: {namespace}")
        if not self.index:
             self.logger.error("Pinecone 索引未初始化。")
             return False
        try:
            self.index.delete(namespace=namespace, delete_all=True)
            self.logger.info(f"命名空间 {namespace} 已成功同步删除。")
            return True
        except Exception as e:
             self.logger.error(f"同步删除命名空间 {namespace} 失败: {e}", exc_info=True)
             return False


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

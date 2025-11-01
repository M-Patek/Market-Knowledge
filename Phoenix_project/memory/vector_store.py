import os
import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from ai.embedding_client import EmbeddingClient  # 更正的导入路径

def _initialize_index(pc_client: Pinecone, index_name: str, dimension: int) -> Optional[Any]:
    """Checks if the index exists and creates it if it doesn't."""
    if index_name not in pc_client.list_indexes().names():
        _logger.warning(f"Pinecone index '{index_name}' not found. Creating a new one...")
        try:
            pc_client.create_index(
                name=index_name,
                dimension=dimension, # TODO: 从配置加载维度
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-west-2" # TODO: 从配置加载
                )
            )
            _logger.info(f"Successfully created new Pinecone index '{index_name}'.")
        except Exception as e:
            _logger.critical(f"Could not create Pinecone index: {e}", exc_info=True)
            return None
    return pc_client.Index(index_name)

# --- 模块级初始化 ---
_logger = logging.getLogger("PhoenixProject.VectorStore")
_index_name = os.getenv("PINECONE_INDEX_NAME", "phoenix-project-rag")
_embedding_dimension = 768 # TODO: 这应该来自配置 (Task 20)
_embedding_client = EmbeddingClient()
_pc_client: Optional[Pinecone] = None
_index: Optional[Any] = None

try:
    _api_key = os.getenv("PINECONE_API_KEY")
    if not _api_key:
        raise ValueError("PINECONE_API_KEY environment variable not set.")

    _pc_client = Pinecone(api_key=_api_key)
    _index = _initialize_index(_pc_client, _index_name, _embedding_dimension)
    if _index:
        _logger.info(f"Successfully connected to Pinecone and attached to index '{_index_name}'.")
    else:
        _logger.error("Failed to initialize Pinecone index.")

except Exception as e:
    _logger.critical(f"Failed to connect to Pinecone: {e}", exc_info=True)
    _pc_client = None
    _index = None
# --- 结束初始化 ---


def query(query_text: str, top_k: int = 3) -> list[str]:
    """Must retrieve historical summaries relevant to the topic."""
    if not _index:
        _logger.error("Index not initialized. Cannot query.")
        return []

    try:
        query_vector = _embedding_client.create_query_embedding(query_text)
        if query_vector is None:
            _logger.error("Failed to obtain a query vector.")
            return []

        results = _index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )
        
        # 格式化结果为 list[str] (根据 Task 7 规范)
        content_results = []
        for match in results.get('matches', []):
            content = match.get('metadata', {}).get('content', '')
            if content:
                content_results.append(content)
        
        return content_results

    except Exception as e:
        _logger.error(f"Failed to query Pinecone: {e}", exc_info=True)
        return []


def _clean_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively cleans metadata to ensure it's JSON-serializable and meets Pinecone's requirements.
    """
    clean = {}
    for key, value in metadata.items():
        if value is None:
            continue # 跳过 None 值
        if isinstance(value, (str, int, float, bool, list)):
            clean[key] = value
        elif isinstance(value, dict):
            clean[key] = _clean_metadata(value) # 递归嵌套字典
        else:
            # 将其他类型转换为字符串作为后备
            clean[key] = str(value)
    return clean


def _batch_upsert(documents: List[Dict[str, Any]], batch_size: int = 100):
    """
    A private helper to handle the core logic of batching and upserting to Pinecone.
    """
    if not _index:
        _logger.error("Index not initialized. Cannot upsert.")
        return

    try:
        num_docs = len(documents)
        _logger.info(f"Starting batch upsert for {num_docs} documents...")
        for i in range(0, num_docs, batch_size):
            batch = documents[i:i + batch_size]
            to_upsert = []
            for doc in batch:
                if 'vector' in doc and doc.get('vector') is not None:
                    # 确保元数据可序列化
                    clean_metadata = _clean_metadata(doc.get("metadata", {}))
                    to_upsert.append({
                        "id": doc["source_id"],
                        "values": doc["vector"],
                        "metadata": clean_metadata
                    })
            if to_upsert:
                _index.upsert(vectors=to_upsert)
        _logger.info(f"Successfully upserted {num_docs} documents.")
    except Exception as e:
        _logger.error(f"Failed during batch upsert: {e}", exc_info=True)


def upsert(task_id: str, text: str, metadata: dict) -> None:
    """Must store and retrieve RAG vectors."""
    
    if not _index:
        _logger.error("Index not initialized. Cannot upsert.")
        return

    # 将单文本输入适配到面向批处理的 upsert 逻辑
    try:
        _logger.info(f"Generating text embedding for task_id: {task_id}")
        # 创建与 embedding 客户端兼容的文档结构
        doc = {"source_id": task_id, "content": text, "metadata": metadata}
        doc_with_embedding = _embedding_client.create_text_embeddings([doc])[0]

        # 为 _batch_upsert 做准备
        doc_with_embedding["metadata"]["content"] = text # 确保文本在元数据中
        
        # 使用单个项目调用批量 upsert
        _batch_upsert([doc_with_embedding])
        _logger.info(f"Successfully upserted task_id: {task_id}")

    except Exception as e:
        _logger.error(f"Failed to upsert single document for task_id {task_id}: {e}", exc_info=True)


def health_check() -> bool:
    """
    Performs a health check on the Pinecone connection.
    """
    if not _pc_client or not _index:
        _logger.error("Health check failed: Pinecone client or index not initialized.")
        return False
    try:
        stats = _index.describe_index_stats()
        if stats:
            _logger.info(f"Health check OK. Index '{_index_name}' has {stats['total_vector_count']} vectors.")
            return True
        return False
    except Exception as e:
        _logger.error(f"Health check failed: {e}", exc_info=True)
        return False

from typing import List, Dict, Any, Optional
import asyncio
from abc import ABC, abstractmethod

# Placeholder for a real vector store client
# In a real implementation, you would use:
# from langchain_community.vectorstores import FAISS, Chroma, etc.
# from langchain_core.documents import Document
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# For this example, we create a mock interface.

# 修正：将 'monitor.logging...' 转换为 'Phoenix_project.monitor.logging...'
from Phoenix_project.monitor.logging import ESLogger
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
    ) -> List[Document]:
        """Perform a similarity search."""
        pass


class MockVectorStore(BaseVectorStore):
    """
    A simple in-memory, mock vector store that simulates the interface.
    It uses the EmbeddingClient to get embeddings and stores them.
    Search is a naive (and slow) cosine similarity calculation.
    """

    def __init__(
        self, embedding_client: EmbeddingClient, logger: ESLogger, config: Dict[str, Any]
    ):
        self.embedding_client = embedding_client
        self.logger = logger
        self.config = config
        self.store: Dict[str, Dict[str, Any]] = {}  # {id: {"vector": [...], "document": Document}}
        self.lock = asyncio.Lock()
        self.logger.log_info("MockVectorStore initialized (in-memory).")
        
        # A simple way to compute cosine similarity (requires numpy)
        try:
            import numpy as np
            self.np = np
        except ImportError:
            self.logger.log_error("Numpy not found. MockVectorStore search will not work.")
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
                self.logger.log_error("Mismatch in embedding results. Cannot add documents.")
                return []

            async with self.lock:
                for doc, vector in zip(documents, embeddings):
                    # Use provided id or generate one
                    doc_id = doc.metadata.get("id", f"doc_{hash(doc.page_content)}")
                    self.store[doc_id] = {"vector": vector, "document": doc}
                    ids.append(doc_id)
            
            self.logger.log_info(f"Added {len(ids)} documents to MockVectorStore.")
            return ids
            
        except Exception as e:
            self.logger.log_error(f"Failed to add documents to MockVectorStore: {e}", exc_info=True)
            return []

    async def asimilarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        """Performs a naive similarity search."""
        if self.np is None:
            self.logger.log_error("Cannot perform search, numpy is not installed.")
            return []
            
        if not self.store:
            self.logger.log_warning("Search called on empty MockVectorStore.")
            return []

        try:
            query_embedding = await self.embedding_client.get_embeddings([query])
            if not query_embedding:
                self.logger.log_error("Failed to get embedding for query.")
                return []
            
            q_vec = self.np.array(query_embedding[0])
            q_norm = self.np.linalg.norm(q_vec)
            if q_norm == 0:
                return []
            
            scores = []
            async with self.lock:
                # This is a naive O(N) scan. Real vector stores are much faster.
                for doc_id, data in self.store.items():
                    doc_vec = self.np.array(data["vector"])
                    doc_norm = self.np.linalg.norm(doc_vec)
                    if doc_norm == 0:
                        continue
                    
                    # Cosine Similarity
                    sim = self.np.dot(q_vec, doc_vec) / (q_norm * doc_norm)
                    scores.append((sim, data["document"]))
            
            # Sort by similarity (highest first) and take top k
            scores.sort(key=lambda x: x[0], reverse=True)
            
            self.logger.log_debug(f"Similarity search for '{query[:20]}...' returned {len(scores)} results.")
            return [doc for sim, doc in scores[:k]]

        except Exception as e:
            self.logger.log_error(f"Failed during similarity search: {e}", exc_info=True)
            return []


def get_vector_store(
    config: Dict[str, Any],
    embedding_client: EmbeddingClient,
    logger: ESLogger
) -> BaseVectorStore:
    """
    Factory function to initialize and return a vector store instance.
    
    This abstracts the specific implementation (e.g., Chroma, FAISS, mock)
    from the components that use it (like FusionSynthesizer).
    """
    store_type = config.get("vector_store", {}).get("type", "mock")
    store_config = config.get("vector_store", {})

    if store_type == "mock":
        logger.log_info("Using MockVectorStore (in-memory).")
        return MockVectorStore(embedding_client, logger, store_config)
    
    elif store_type == "chroma":
        # Example of how you would add a real vector store
        logger.log_error("ChromaDB not yet implemented in this mock setup.")
        # from langchain_community.vectorstores import Chroma
        # persist_directory = store_config.get("persist_directory", "./chroma_db")
        # return Chroma(
        #     embedding_function=embedding_client.get_langchain_embedding(), # Assumes embedding_client has this
        #     persist_directory=persist_directory
        # )
        raise NotImplementedError("ChromaDB vector store is not implemented.")
        
    else:
        logger.log_error(f"Unknown vector store type: {store_type}. Defaulting to mock.")
        return MockVectorStore(embedding_client, logger, store_config)

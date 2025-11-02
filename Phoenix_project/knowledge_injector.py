import asyncio
from typing import Dict, Any, List

# 修复：将相对导入 'from .monitor.logging...' 更改为绝对导入
from monitor.logging import get_logger
# 修复：将相对导入 'from .memory.vector_store...' 更改为绝对导入
from memory.vector_store import VectorStore
# from .knowledge_graph_service import KnowledgeGraphService
# from .ai.relation_extractor import RelationExtractor

logger = get_logger(__name__)

class KnowledgeInjector:
    """
    A service responsible for processing unstructured documents (e.g.,
    SEC 10-K filings, long articles) and "injecting" the
    knowledge into the system's memories (VectorStore, KG).
    """

    def __init__(self, config: Dict[str, Any], vector_store: VectorStore):
        self.config = config
        self.vector_store = vector_store
        self.chunk_size = config.get("chunk_size", 1200)
        self.chunk_overlap = config.get("chunk_overlap", 200)
        logger.info("KnowledgeInjector initialized.")

    async def ingest_documents(self, docs: List[Dict[str, Any]]):
        """
        Ingests documents into vector memory (and optionally KG).
        docs: list of dict with fields:
          - id: str
          - text: str
          - metadata: dict
        """
        logger.info("Ingesting %d documents ...", len(docs))
        for doc in docs:
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            chunks = self._split_text(text)

            # Persist to vector store
            for chunk in chunks:
                await self.vector_store.add_texts([chunk], [metadata])

        logger.info("Ingest complete.")

    def _split_text(self, text: str) -> List[str]:
        """
        Naive splitter by character window with overlap.
        A better implementation would use a library like 'langchain'
        or 'nltk' to split on sentences or paragraphs.
        """
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])

            start += self.chunk_size - self.chunk_overlap
            if end >= len(text):
                break

        return chunks

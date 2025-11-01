import asyncio
from typing import Dict, Any, List

from ..monitor.logging import get_logger
from ..memory.vector_store import VectorStore
# from .knowledge_graph_service import KnowledgeGraphService
# from .ai.relation_extractor import RelationExtractor

logger = get_logger(__name__)

class KnowledgeInjector:
    """
    A service responsible for processing unstructured documents (e.g.,
    SEC 10-K filings, long articles) and "injecting" the
    knowledge into the system's memories (VectorStore, KG).
    
    This is typically run as a batch process, not in the real-time loop.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        vector_store: VectorStore
        # kg_service: KnowledgeGraphService,
        # relation_extractor: RelationExtractor
    ):
        """
        Initializes the KnowledgeInjector.
        
        Args:
            config: Main system configuration.
            vector_store: The vector database client.
            # kg_service: The knowledge graph client.
            # relation_extractor: The AI agent that finds relationships.
        """
        self.config = config.get('knowledge_injector', {})
        self.vector_store = vector_store
        # self.kg_service = kg_service
        # self.relation_extractor = relation_extractor
        
        # Text chunking parameters
        self.chunk_size = self.config.get('chunk_size', 1000)
        self.chunk_overlap = self.config.get('chunk_overlap', 100)
        
        logger.info("KnowledgeInjector initialized.")

    async def process_document(self, document_text: str, metadata: Dict[str, Any]):
        """
        Processes a single large document.
        
        Args:
            document_text (str): The full text of the document.
            metadata (Dict[str, Any]): Metadata about the document (e.g.,
                                       source_url, document_id, symbols).
        """
        logger.info(f"Processing document: {metadata.get('document_id', 'Unknown')}")
        
        # 1. Chunk the document
        text_chunks = self._chunk_text(document_text)
        logger.info(f"Split document into {len(text_chunks)} chunks.")
        
        # 2. Inject chunks into VectorStore
        # (This also handles embedding creation via the VectorStore's client)
        chunk_ids = [f"{metadata.get('document_id', 'doc')}_chunk_{i}" for i in range(len(text_chunks))]
        
        # We add the document-level metadata to each chunk
        chunk_metadatas = []
        for i, chunk in enumerate(text_chunks):
            chunk_meta = metadata.copy()
            chunk_meta.update({
                "chunk_index": i,
                "text_snippet": chunk[:150] # For preview
            })
            chunk_metadatas.append(chunk_meta)
            
        try:
            await self.vector_store.upsert_batch(
                ids=chunk_ids,
                texts=text_chunks,
                metadatas=chunk_metadatas
            )
            logger.info(f"Successfully upserted {len(text_chunks)} chunks to VectorStore.")
        except Exception as e:
            logger.error(f"Failed to upsert chunks to VectorStore: {e}", exc_info=True)
            return

        # 3. (Future) Extract and Inject Knowledge Graph Relations
        # This is computationally expensive
        # try:
        #     relations = await self.relation_extractor.extract_relations(document_text)
        #     logger.info(f"Extracted {len(relations)} relations from document.")
        #     for rel in relations:
        #         await self.kg_service.add_relationship(rel['entity_a'], rel['entity_b'], rel['relationship'])
        #     logger.info("Successfully injected relations into Knowledge Graph.")
        # except Exception as e:
        #     logger.error(f"Failed to extract/inject KG relations: {e}", exc_info=True)
            
    def _chunk_text(self, text: str) -> List[str]:
        """
        A simple, non-semantic text chunker.
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

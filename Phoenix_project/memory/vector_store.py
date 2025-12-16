import os
import logging
import asyncio
import uuid
from typing import List, Dict, Any, Optional
# [P0-INFRA-03] Replace with AsyncQdrantClient
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Manages embedding storage and retrieval using Qdrant (Async).
    Stores L1 Evidence and RAG Documents.
    """

    def __init__(self, config: Dict[str, Any], embedding_client: Any, vector_size: Optional[int] = None):
        self.config = config
        self.embedding_client = embedding_client
        
        self.host = self.config.get("host", "phoenix_qdrant")
        self.port = self.config.get("port", 6333)
        self.collection_name = self.config.get("l1_evidence_collection", "phoenix_l1_evidence")
        
        # [Task FIX-CRIT-004] Priority: Explicit Arg > Config > Default
        if vector_size:
            self.vector_size = vector_size
        else:
            self.vector_size = self.config.get("vector_size", 1536)
        
        logger.info(f"VectorStore configured with vector_size: {self.vector_size}")
        
        # [P0-INFRA-03] Async Client Initialization
        self.client = AsyncQdrantClient(host=self.host, port=self.port)
        self._initialized = False

    async def _ensure_initialized(self):
        """
        [P0-INFRA-03] Lazy async initialization of collection.
        """
        if self._initialized:
            return

        try:
            # Check if collection exists (Async)
            try:
                await self.client.get_collection(self.collection_name)
            except Exception:
                logger.info(f"Collection {self.collection_name} not found. Creating...")
                await self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE)
                )
            
            self._initialized = True
            logger.info(f"VectorStore initialized for {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to initialize VectorStore collection: {e}")

    async def add(self, texts: List[str], metadatas: List[Dict[str, Any]], ids: Optional[List[str]] = None) -> bool:
        """
        Embeds texts and stores them in the vector database (Async).
        """
        if not self.client:
            return False
            
        try:
            await self._ensure_initialized()
            
            # 1. Generate Embeddings 
            # [P0-INFRA-03] Offload potentially blocking embedding call to executor
            loop = asyncio.get_running_loop()
            embeddings = await loop.run_in_executor(None, self.embedding_client.embed_batch, texts)
            
            if not embeddings:
                logger.warning("No embeddings generated.")
                return False
            
            # 2. Prepare Points
            points = []
            
            for i, text in enumerate(texts):
                point_id = ids[i] if ids else str(uuid.uuid4())
                vector = embeddings[i]
                payload = metadatas[i]
                payload["text"] = text # Store original text
                
                points.append(models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload
                ))
            
            # 3. Upsert (Async)
            await self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            return True
            
        except Exception as e:
            logger.error(f"VectorStore add failed: {e}")
            return False

    async def search(self, query_text: str, limit: int = 5, filter_criteria: Dict = None) -> List[Dict[str, Any]]:
        """
        Semantic search for query_text (Async).
        """
        if not self.client:
            return []
            
        try:
            await self._ensure_initialized()

            # 1. Embed Query
            loop = asyncio.get_running_loop()
            query_vector = await loop.run_in_executor(None, self.embedding_client.embed_query, query_text)
            
            if not query_vector:
                return []
            
            # 2. Search (Async)
            search_result = await self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=limit
            )
            
            # 3. Format Results
            results = []
            for hit in search_result:
                results.append({
                    "id": hit.id,
                    "score": hit.score,
                    "content": hit.payload.get("text", ""),
                    "metadata": hit.payload
                })
                
            return results

        except Exception as e:
            logger.error(f"VectorStore search failed: {e}")
            return []

    async def close(self):
        """[Fix] Resource Cleanup: Close vector database client."""
        if hasattr(self, 'client') and hasattr(self.client, 'close'):
            try:
                await self.client.close()
            except Exception:
                pass

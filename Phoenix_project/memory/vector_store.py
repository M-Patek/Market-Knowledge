import os
import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Manages embedding storage and retrieval using Qdrant (default) or generic interface.
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
        
        self.client = None
        self._initialize_client()

    def _initialize_client(self):
        try:
            # Initialize Qdrant Client
            # Use host/port
            self.client = QdrantClient(host=self.host, port=self.port)
            
            # Ensure Collection Exists
            # Check if collection exists, if not create
            try:
                self.client.get_collection(self.collection_name)
            except Exception:
                logger.info(f"Collection {self.collection_name} not found. Creating...")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(size=self.vector_size, distance=models.Distance.COSINE)
                )
            
            logger.info(f"VectorStore initialized for {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to initialize VectorStore: {e}")
            self.client = None

    def add(self, texts: List[str], metadatas: List[Dict[str, Any]], ids: Optional[List[str]] = None) -> bool:
        """
        Embeds texts and stores them in the vector database.
        """
        if not self.client:
            return False
            
        try:
            # 1. Generate Embeddings
            embeddings = self.embedding_client.embed_batch(texts)
            if not embeddings:
                logger.warning("No embeddings generated.")
                return False
            
            # 2. Prepare Points
            points = []
            import uuid
            
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
            
            # 3. Upsert
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            return True
            
        except Exception as e:
            logger.error(f"VectorStore add failed: {e}")
            return False

    def search(self, query_text: str, limit: int = 5, filter_criteria: Dict = None) -> List[Dict[str, Any]]:
        """
        Semantic search for query_text.
        """
        if not self.client:
            return []
            
        try:
            # 1. Embed Query
            query_vector = self.embedding_client.embed_query(query_text)
            if not query_vector:
                return []
            
            # 2. Search
            # Simple search without complex filters for now
            search_result = self.client.search(
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

    def close(self):
        """[Fix] Resource Cleanup: Close vector database client."""
        # QdrantClient (HTTP) might not have a strict close, but good practice to allow for it
        # if using gRPC or persistent connections.
        if hasattr(self, 'client') and hasattr(self.client, 'close'):
            try:
                self.client.close()
            except Exception:
                pass

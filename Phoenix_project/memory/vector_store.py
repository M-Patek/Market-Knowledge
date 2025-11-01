from pinecone import Pinecone, ServerlessSpec, PodSpec
from typing import Dict, Any, List, Optional
import os

from ..monitor.logging import get_logger
from ..ai.embedding_client import EmbeddingClient
from ..core.schemas.data_schema import MarketEvent

logger = get_logger(__name__)

class VectorStore:
    """
    Manages all interactions with the vector database (Pinecone).
    This includes upserting (embedding + storing) and searching.
    """

    def __init__(self, config: Dict[str, Any], embedding_client: EmbeddingClient):
        """
        Initializes the VectorStore client.
        
        Args:
            config: Main configuration. Expects 'vector_store' block.
            embedding_client: The client to use for generating embeddings.
        """
        self.config = config.get('vector_store', {})
        self.embedding_client = embedding_client
        
        self.api_key = os.environ.get("PINECONE_API_KEY")
        if not self.api_key:
            logger.error("PINECONE_API_KEY environment variable not set.")
            raise ValueError("PINECONE_API_KEY not set")
            
        self.pc = Pinecone(api_key=self.api_key)
        
        self.index_name = self.config.get('index_name', 'phoenix-market-knowledge')
        self.dimension = self.embedding_client.dimensions
        
        self.index = self._get_or_create_index()

    def _get_or_create_index(self):
        """Connects to the Pinecone index, creating it if it doesn't exist."""
        
        # Check if index exists
        if self.index_name not in [idx.get('name') for idx in self.pc.list_indexes()]:
            logger.warning(f"Pinecone index '{self.index_name}' not found. Creating...")
            try:
                # Get environment (e.g., "gcp-starter")
                cloud_env = self.config.get('cloud_env', 'gcp-starter')
                
                if cloud_env == 'gcp-starter':
                    # Use serverless for free tier
                    self.pc.create_index(
                        name=self.index_name,
                        dimension=self.dimension,
                        metric=self.config.get('metric', 'cosine'),
                        spec=ServerlessSpec(cloud='gcp', region='us-central1') # Example
                    )
                else:
                    # Use pod-based for paid tiers
                    self.pc.create_index(
                        name=self.index_name,
                        dimension=self.dimension,
                        metric=self.config.get('metric', 'cosine'),
                        spec=PodSpec(environment=cloud_env)
                    )
                
                logger.info(f"Successfully created Pinecone index: {self.index_name}")
            except Exception as e:
                logger.error(f"Failed to create Pinecone index: {e}", exc_info=True)
                raise
        else:
            logger.info(f"Connected to existing Pinecone index: {self.index_name}")
            
        return self.pc.Index(self.index_name)

    async def upsert_event(self, event: MarketEvent) -> bool:
        """
        Embeds and upserts a single MarketEvent.
        
        Args:
            event (MarketEvent): The event to upsert.
            
        Returns:
            bool: True on success, False on failure.
        """
        
        # 1. Create the text to be embedded
        text_to_embed = f"Headline: {event.headline}\nSummary: {event.summary}"
        
        # 2. Get embedding
        embedding = await self.embedding_client.get_embedding(
            text_to_embed, 
            cache_key=event.event_id
        )
        
        if not embedding:
            logger.error(f"Failed to get embedding for event {event.event_id}. Skipping upsert.")
            return False
            
        # 3. Prepare metadata
        # Pinecone metadata must be flat (no nested dicts)
        metadata = {
            "source": event.source,
            "headline": event.headline,
            "url": event.url,
            "symbols": ",".join(event.symbols), # Convert list to comma-separated string
            "event_timestamp": event.timestamp.isoformat()
            # ... add other flat metadata ...
        }
        
        # 4. Upsert to Pinecone
        try:
            self.index.upsert(
                vectors=[
                    {
                        "id": event.event_id,
                        "values": embedding,
                        "metadata": metadata
                    }
                ],
                namespace=self.config.get('namespace', 'events')
            )
            return True
        except Exception as e:
            logger.error(f"Failed to upsert event {event.event_id} to Pinecone: {e}", exc_info=True)
            return False

    async def upsert_batch(self, ids: List[str], texts: List[str], metadatas: List[Dict]) -> bool:
        """
        Embeds and upserts a batch of texts (e.g., document chunks).
        
        Args:
            ids (List[str]): List of unique IDs.
            texts (List[str]): List of corresponding texts to embed.
            metadatas (List[Dict]): List of corresponding (flat) metadata dicts.
            
        Returns:
            bool: True on success, False on failure.
        """
        
        # 1. Get embeddings in batch
        embeddings = await self.embedding_client.get_embeddings(texts, cache_keys=ids)
        
        vectors_to_upsert = []
        for i, embedding in enumerate(embeddings):
            if embedding:
                vectors_to_upsert.append({
                    "id": ids[i],
                    "values": embedding,
                    "metadata": metadatas[i] # Assume metadata is already flat
                })
            else:
                logger.warning(f"Skipping upsert for id {ids[i]}: Failed to get embedding.")

        if not vectors_to_upsert:
            return False

        # 2. Upsert in batches (Pinecone has a limit, e.g., 100)
        batch_size = 100
        try:
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i : i + batch_size]
                self.index.upsert(
                    vectors=batch,
                    namespace=self.config.get('namespace_docs', 'documents')
                )
            logger.info(f"Successfully upserted {len(vectors_to_upsert)} vectors in batches.")
            return True
        except Exception as e:
            logger.error(f"Failed to batch upsert to Pinecone: {e}", exc_info=True)
            return False

    async def search(
        self, 
        query_embedding: List[float], 
        top_k: int, 
        filter_dict: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Searches the vector store.
        
        Args:
            query_embedding (List[float]): The embedding vector of the query.
            top_k (int): Number of results to return.
            filter_dict (Optional[Dict]): Pinecone metadata filter.
            
        Returns:
            List[Dict[str, Any]]: A list of search results.
        """
        
        try:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                filter=filter_dict,
                include_metadata=True,
                namespace=self.config.get('namespace', 'events') # Search events namespace by default
            )
            
            # Format results
            formatted_results = []
            for match in results.get('matches', []):
                formatted_results.append({
                    "id": match.get('id'),
                    "score": match.get('score'),
                    "metadata": match.get('metadata', {}),
                    "text": match.get('metadata', {}).get('headline', '') # Add text snippet
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Failed to search Pinecone: {e}", exc_info=True)
            return []

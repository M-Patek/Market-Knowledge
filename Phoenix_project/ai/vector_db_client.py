import os
import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from .embedding_client import EmbeddingClient

class VectorDBClient:
    """
    A client to manage interactions with the Pinecone vector database.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the connection to Pinecone and the embedding client.

        Args:
            config: A dictionary containing configuration for the vector DB,
                    including 'index_name' and 'embedding_dimension'.
        """
        self.logger = logging.getLogger("PhoenixProject.VectorDBClient")
        self.config = config
        self.index_name = self.config.get("index_name", "phoenix-project-rag")
        self.pc: Optional[Pinecone] = None
        self.index = None
        self.embedding_client = EmbeddingClient()

        try:
            api_key = os.getenv("PINECONE_API_KEY")
            if not api_key:
                raise ValueError("PINECONE_API_KEY environment variable not set.")

            self.pc = Pinecone(api_key=api_key)
            self._initialize_index()
            self.logger.info(f"Successfully connected to Pinecone and attached to index '{self.index_name}'.")

        except Exception as e:
            self.logger.critical(f"Failed to connect to Pinecone: {e}", exc_info=True)
            self.pc = None
            self.index = None

    def _initialize_index(self):
        """Checks if the index exists and creates it if it doesn't."""
        if not self.pc: return

        if self.index_name not in self.pc.list_indexes().names():
            self.logger.warning(f"Pinecone index '{self.index_name}' not found. Creating a new one...")
            try:
                dimension = self.config.get("embedding_dimension")
                if not dimension:
                    raise ValueError("'embedding_dimension' must be set in the vector_database config.")

                # and a serverless spec for cost-effective scaling.
                self.pc.create_index(
                    name=self.index_name,
                    dimension=dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-west-2"
                    )
                )
                self.logger.info(f"Successfully created new Pinecone index '{self.index_name}'.")
            except Exception as e:
                self.logger.critical(f"Could not create Pinecone index: {e}", exc_info=True)
                raise
        
        self.index = self.pc.Index(self.index_name)

    def query(self, query: str, top_k: int = 10, namespace: Optional[str] = None, query_vector: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        """
        Queries the vector database with a given text string.
        Can optionally accept a pre-computed vector (e.g., from HyDE).
        """
        if not self.index:
            self.logger.error("Index not initialized. Cannot query.")
            return []

        try:
            if query_vector is None:
                query_vector = self.embedding_client.create_query_embedding(query)
            if query_vector is None:
                self.logger.error("Failed to obtain a query vector.")
                return []

            results = self.index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                namespace=namespace
            )
            
            # Format results into a more usable list of dictionaries
            formatted_results = []
            for match in results.get('matches', []):
                res = {
                    "source_id": match.get('id'),
                    "vector_similarity_score": match.get('score'),
                    "metadata": match.get('metadata', {}),
                    "content": match.get('metadata', {}).get('content', '') # Assuming content is stored in metadata
                }
                formatted_results.append(res)

            return formatted_results

        except Exception as e:
            self.logger.error(f"Failed to query Pinecone: {e}", exc_info=True)
            return []

    def _batch_upsert(self, documents: List[Dict[str, Any]], batch_size: int = 100):
        """
        A private helper to handle the core logic of batching and upserting to Pinecone.
        """
        if not self.index:
            self.logger.error("Index not initialized. Cannot upsert.")
            return

        try:
            num_docs = len(documents)
            self.logger.info(f"Starting batch upsert for {num_docs} documents...")
            for i in range(0, num_docs, batch_size):
                batch = documents[i:i + batch_size]
                to_upsert = []
                for doc in batch:
                    if 'vector' in doc and doc.get('vector') is not None:
                        # Ensure metadata is serializable
                        clean_metadata = self._clean_metadata(doc.get("metadata", {}))
                        to_upsert.append({
                            "id": doc["source_id"],
                            "values": doc["vector"],
                            "metadata": clean_metadata
                        })
                if to_upsert:
                    self.index.upsert(vectors=to_upsert)
            self.logger.info(f"Successfully upserted {num_docs} documents.")
        except Exception as e:
            self.logger.error(f"Failed during batch upsert: {e}", exc_info=True)

    def upsert(self, documents: List[Dict[str, Any]], batch_size: int = 100):
        """
        Upserts a list of documents. Generates text embeddings if they are missing.
        """
        docs_to_embed = [doc for doc in documents if 'vector' not in doc]
        docs_with_vector = [doc for doc in documents if 'vector' in doc]

        if docs_to_embed:
            self.logger.info(f"Generating text embeddings for {len(docs_to_embed)} documents.")
            docs_to_embed = self.embedding_client.create_text_embeddings(docs_to_embed)

        all_documents = docs_with_vector + docs_to_embed
        self._batch_upsert(all_documents, batch_size)

    def upsert_time_series(self, documents: List[Dict[str, Any]], batch_size: int = 100):
        """Upserts time-series data after generating their embeddings."""
        self.logger.info(f"Generating time-series embeddings for {len(documents)} documents before upsert.")
        documents_with_embeddings = self.embedding_client.create_time_series_embeddings(documents)
        self._batch_upsert(documents_with_embeddings, batch_size)

    def upsert_tabular_embeddings(self, documents: List[Dict[str, Any]], batch_size: int = 100):
        """Upserts tabular data after generating their embeddings."""
        self.logger.info(f"Generating tabular embeddings for {len(documents)} documents before upsert.")
        documents_with_embeddings = self.embedding_client.create_tabular_embeddings(documents)
        self._batch_upsert(documents_with_embeddings, batch_size)

    def _clean_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively cleans metadata to ensure it's JSON-serializable and meets Pinecone's requirements.
        """
        clean = {}
        for key, value in metadata.items():
            if value is None:
                continue # Skip None values
            if isinstance(value, (str, int, float, bool, list)):
                clean[key] = value
            elif isinstance(value, dict):
                clean[key] = self._clean_metadata(value) # Recurse for nested dicts
            else:
                # Convert other types to string as a fallback
                clean[key] = str(value)
        return clean

    def health_check(self) -> bool:
        """
        Performs a health check on the Pinecone connection.
        """
        if not self.pc or not self.index:
            self.logger.error("Health check failed: Pinecone client or index not initialized.")
            return False
        try:
            stats = self.index.describe_index_stats()
            if stats:
                self.logger.info(f"Health check OK. Index '{self.index_name}' has {stats['total_vector_count']} vectors.")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Health check failed: {e}", exc_info=True)
            return False

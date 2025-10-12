# ai/vector_db_client.py
"""
Manages the connection and lifecycle of the vector database index (Pinecone).
This client is responsible for initialization, health checks, and creation of the
index if it does not already exist.
"""
import os
import logging
from pinecone import Pinecone, ServerlessSpec
from typing import Optional, List, Dict, Any
from tenacity import retry, wait_fixed, stop_after_attempt

from .embedding_client import EmbeddingClient

class VectorDBClient:
    """
    A client to manage interactions with the Pinecone vector database.
    """
    def __init__(self, index_name: str = "phoenix-project-rag"):
        """
        Initializes the connection to Pinecone and the embedding client.

        Args:
            index_name: The name of the Pinecone index to use.
        """
        self.logger = logging.getLogger("PhoenixProject.VectorDBClient")
        self.index_name = index_name
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
            self.logger.error(f"Failed to initialize Pinecone Vector DB Client: {e}")
            # The client will be in a non-operational state if initialization fails.
            self.pc = None
            self.index = None

    def _initialize_index(self):
        """Checks if the index exists and creates it if it doesn't."""
        if not self.pc: return

        if self.index_name not in self.pc.list_indexes().names():
            self.logger.warning(f"Pinecone index '{self.index_name}' not found. Creating a new one...")
            try:
                # We specify the dimension for OpenAI's text-embedding-3-large model
                # and a serverless spec for cost-effective scaling.
                self.pc.create_index(
                    name=self.index_name,
                    dimension=768, # Gemini-embedding-001 uses a dimension of 768
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-west-2"
                    )
                )
                self.logger.info(f"Successfully created new Pinecone index '{self.index_name}'.")
            except Exception as e:
                self.logger.error(f"Failed to create Pinecone index: {e}")
                raise
        
        self.index = self.pc.Index(self.index_name)

    def is_healthy(self) -> bool:
        """
        Checks if the client is properly initialized and connected.
        """
        if not self.index:
            return False
        try:
            # A lightweight check to see if the index is responsive
            stats = self.index.describe_index_stats()
            self.logger.debug(f"Pinecone index is healthy. Vector count: {stats.get('total_vector_count', 0)}")
            return True
        except Exception as e:
            self.logger.error(f"Pinecone health check failed: {e}")
            return False

    @retry(wait=wait_fixed(2), stop=stop_after_attempt(3))
    def upsert(self, documents: List[Dict[str, Any]], batch_size: int = 100):
        """
        Generates embeddings and upserts documents into the Pinecone index.

        Args:
            documents: A list of documents to process. Each document is a dictionary.
            batch_size: The number of documents to process in each batch.
        """
        if not self.is_healthy() or not self.index:
            self.logger.error("Cannot upsert, VectorDBClient is not healthy.")
            return

        try:
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                self.logger.info(f"Processing batch {i // batch_size + 1} for upsert...")

                # 1. Generate embeddings for the batch
                docs_with_vectors = self.embedding_client.create_embeddings(batch)
                if not docs_with_vectors:
                    self.logger.warning("Embedding generation returned no vectors. Skipping batch.")
                    continue

                # 2. Prepare records for Pinecone upsert
                records_to_upsert = []
                for doc in docs_with_vectors:
                    # Pinecone requires an 'id' and 'values' (the vector).
                    # 'metadata' should contain everything else we want to store and filter on.
                    metadata = {k: v for k, v in doc.items() if k not in ['vector', 'content']}
                    metadata['content_snippet'] = doc.get('content', '')[:500] # Store a snippet
                    
                    # [CRITICAL] Ensure the model version from the embedding client is in the metadata
                    if 'embedding_model_version' not in metadata:
                        self.logger.warning(f"Document '{doc.get('source_id')}' is missing embedding_model_version. This should not happen.")
                        metadata['embedding_model_version'] = 'unknown'
                    
                    records_to_upsert.append({
                        "id": doc['source_id'],
                        "values": doc['vector'],
                        "metadata": metadata
                    })

                # 3. Upsert the batch to the index
                self.index.upsert(vectors=records_to_upsert)
                self.logger.info(f"Successfully upserted {len(records_to_upsert)} vectors.")

        except Exception as e:
            self.logger.error(f"An error occurred during the upsert process: {e}")
            raise

    def upsert_time_series(self, documents: List[Dict[str, Any]], batch_size: int = 50):
        """
        Generates and upserts time-series embeddings into the Pinecone index.

        Args:
            documents: A list of documents containing time-series data.
            batch_size: The number of documents to process in each batch.
        """
        if not self.is_healthy() or not self.index:
            self.logger.error("Cannot upsert, VectorDBClient is not healthy.")
            return

        try:
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                self.logger.info(f"Processing time-series batch {i // batch_size + 1} for upsert...")

                # 1. Generate embeddings using the specialized TS2Vec method
                docs_with_vectors = self.embedding_client.create_time_series_embeddings(batch)
                if not docs_with_vectors:
                    continue

                # 2. Prepare records for Pinecone upsert
                records_to_upsert = [
                    {
                        "id": doc['source_id'],
                        "values": doc['vector']，
                        "metadata": {k: v for k, v in doc.items() if k not in ['vector', 'time_series_data']}
                    }
                    for doc in docs_with_vectors if 'vector' in doc
                ]

                # 3. Upsert the batch to the index
                if records_to_upsert:
                    self.index.upsert(vectors=records_to_upsert)
                    self.logger.info(f"Successfully upserted {len(records_to_upsert)} time-series vectors.")

        except Exception as e:
            self.logger.error(f"An error occurred during the time-series upsert process: {e}")
            raise

    def upsert_tabular_embeddings(self, documents: List[Dict[str, Any]], batch_size: int = 100):
        """
        Generates and upserts tabular data embeddings into the Pinecone index.

        Args:
            documents: A list of documents representing structured table rows.
            batch_size: The number of documents to process in each batch.
        """
        if not self.is_healthy() or not self.index:
            self.logger.error("Cannot upsert, VectorDBClient is not healthy.")
            return

        try:
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                self.logger.info(f"Processing tabular data batch {i // batch_size + 1} for upsert...")

                # 1. Generate embeddings using the specialized tabular-to-text method
                docs_with_vectors = self.embedding_client.create_tabular_embeddings(batch)
                if not docs_with_vectors:
                    continue

                # 2. Prepare and upsert records
                records_to_upsert = [
                    {"id": doc['source_id'], "values": doc['vector'], "metadata": doc}
                    for doc in docs_with_vectors if 'vector' in doc
                ]
                if records_to_upsert:
                    self.index.upsert(vectors=records_to_upsert)
                    self.logger.info(f"Successfully upserted {len(records_to_upsert)} tabular vectors.")
        except Exception as e:
            self.logger.error(f"An error occurred during the tabular upsert process: {e}")
            raise

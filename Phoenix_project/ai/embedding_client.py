# ai/embedding_client.py
import os
import logging
from typing import List, Dict, Any, Optional
import google.generativeai as genai
import numpy as np

class EmbeddingClient:
    """
    A client to handle the creation of text and other embeddings using a configured AI provider.
    """
    def __init__(self, provider: str = 'google', model_name: str = 'text-embedding-004', batch_size: int = 100):
        self.logger = logging.getLogger("PhoenixProject.EmbeddingClient")
        self.provider = provider
        self.model_name = model_name
        self.batch_size = batch_size
        self._configure_client()

    def _configure_client(self):
        """Configures the generative AI client based on the specified provider."""
        if self.provider == 'google':
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY environment variable not set for Google provider.")
            genai.configure(api_key=api_key)
            self.logger.info(f"EmbeddingClient configured for Google with model '{self.model_name}'.")
        else:
            raise ValueError(f"Unsupported AI provider: {self.provider}")

    def create_text_embeddings(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Creates text embeddings for a list of documents.

        Args:
            documents: A list of dictionaries, where each dict represents a document
                       and must contain a 'content' key.

        Returns:
            The same list of documents, with a 'vector' key added to each dictionary,
            containing the document's embedding.
        """
        try:
            texts_to_embed = [doc['content'] for doc in documents]
            # The genai library automatically handles batching.
            result = genai.embed_content(
                model=self.model_name,
                content=texts_to_embed,
                task_type="RETRIEVAL_DOCUMENT"
            )
            embeddings = result['embedding']

            for doc, embedding in zip(documents, embeddings):
                doc['vector'] = embedding
            
            self.logger.info(f"Successfully created text embeddings for {len(documents)} documents.")
            return documents
        except Exception as e:
            self.logger.error(f"Failed to create text embeddings: {e}")
            # Return documents without vectors so the pipeline isn't completely halted.
            self.logger.warning("Returning documents without text vectors due to an error.")
            return documents

    def create_query_embedding(self, query: str) -> Optional[List[float]]:
        """Creates an embedding for a single query string."""
        try:
            result = genai.embed_content(
                model=self.model_name,
                content=query,
                task_type="RETRIEVAL_QUERY"
            )
            return result['embedding']
        except Exception as e:
            self.logger.error(f"Failed to create query embedding: {e}")
            return None

    def create_time_series_embeddings(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Creates embeddings for time-series data.
        This is a placeholder for a real time-series embedding model like TS2Vec.
        """
        try:
            # In a real system, you would load a model trained on financial data.
            # For this implementation, we use a pre-trained general model.
            # The model is loaded once and cached in memory for efficiency.
            if not hasattr(self, '_ts2vec_model'): 
                # This path is a placeholder. A real implementation would download or load a trained model.
                self.logger.warning("Loading a placeholder TS2Vec model. For production, use a trained model.")
                # self._ts2vec_model = TS2Vec.load_from_checkpoint('path/to/pretrained/ts2vec/model.ckpt')
                # For now, we create a dummy model that outputs random vectors of the correct dimension
                class DummyTS2Vec:
                    def encode(self, data, encoding_window):
                        return np.random.rand(data.shape[0], 320) # A common TS2Vec output dimension
                self._ts2vec_model = DummyTS2Vec()
 
            for doc in documents:
                ts_data = doc.get("time_series_data")
                if isinstance(ts_data, np.ndarray):
                    # TS2Vec expects a 3D array (batch, timestamp, feature)
                    if ts_data.ndim == 2:
                        ts_data = np.expand_dims(ts_data, axis=0)
                    
                    # Generate the embedding for the time series
                    embedding = self._ts2vec_model.encode(ts_data, encoding_window='full_series')
                    doc['vector'] = embedding.flatten().tolist()
                else:
                    self.logger.warning(f"Document '{doc.get('source_id')}' is missing valid 'time_series_data'.")

            return documents
        except Exception as e:
            self.logger.error(f"Failed to create time-series embeddings: {e}")
            self.logger.warning("Returning documents without time-series vectors due to an error.")
            return documents

    def _convert_tabular_to_text(self, tabular_row: Dict[str, Any]) -> str:
        """Converts a dictionary of tabular data into a single descriptive string."""
        # Simple implementation: join key-value pairs.
        # A more sophisticated approach would use templates or natural language generation.
        return ". ".join([f"{key.replace('_', ' ')} is {value}" for key, value in tabular_row.items()])

    def create_tabular_embeddings(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Creates embeddings for tabular data by first converting it to a text representation.
        """
        try:
            # Create a textual representation for each row of tabular data.
            text_representations = [self._convert_tabular_to_text(doc.get("tabular_data", {})) for doc in documents]
            
            # Embed the textual representations in a batch.
            result = genai.embed_content(
                model=self.model_name,
                content=text_representations,
                task_type="RETRIEVAL_DOCUMENT"
            )
            embeddings = result['embedding']

            for doc, embedding in zip(documents, embeddings):
                doc['vector'] = embedding

            self.logger.info(f"Successfully created tabular embeddings for {len(documents)} documents.")
            return documents
        except Exception as e:
            self.logger.error(f"Failed to create tabular embeddings: {e}")
            self.logger.warning("Returning documents without tabular vectors due to an error.")
            return documents

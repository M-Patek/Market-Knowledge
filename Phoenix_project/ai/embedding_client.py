# ai/embedding_client.py
"""
Manages the creation of vector embeddings from text documents using a
third-party service like OpenAI.
"""
import os
import json
import logging
import google.generativeai as genai
import numpy as np
from ts2vec import TS2Vec
from typing import List, Dict, Any, Union
from tenacity import retry, wait_random_exponential, stop_after_attempt

class EmbeddingClient:
    """
    A client for generating text embeddings using the OpenAI API.
    """
    def __init__(self, model_name: str = "gemini-embedding-001"):
        """
        Initializes the OpenAI client.

        Args:
            model_name: The name of the embedding model to use.
        """
        self.logger = logging.getLogger("PhoenixProject.EmbeddingClient")
        self.model_name = model_name
        self.model_version_info = None
        
        try:
            # Load the version info for the specified model
            # In a real system, model names might have versions like 'model_name:1'
            model_version_path = f"models/embedding_models/{self.model_name}_v1.json"
            with open(model_version_path, 'r', encoding='utf-8') as f:
                self.model_version_info = json.load(f)
            self.logger.info(f"Loaded model version info: {self.model_version_info.get('version')}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.logger.error(f"Could not load model version info from '{model_version_path}': {e}")

        try:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set.")
            
            genai.configure(api_key=api_key)
            self.logger.info(f"EmbeddingClient initialized with model '{self.model_name}'.")

        except Exception as e:
            self.logger.error(f"Failed to initialize GenerativeAI client for embeddings: {e}")

    @retry(wait=wait_random_exponential(multiplier=1, min=2, max=30), stop=stop_after_attempt(3))
    def create_embeddings(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generates embeddings for a batch of documents.

        Args:
            documents: A list of documents, where each document is a dictionary
                       expected to have a 'content' key.

        Returns:
            A list of the original documents, each updated with a 'vector' key.
        """
        texts = [doc.get('content', '') for doc in documents]
        if not texts:
            return []

        try:
            result = genai.embed_content(
                model=f"models/{self.model_name}",
                content=texts,
                task_type="retrieval_document"
            )
            
            # Attach the generated vector to each original document object
            for doc, embedding in zip(documents, result['embedding']):
                doc['vector'] = embedding
                # [NEW] Tag the document with the model version
                if self.model_version_info:
                    doc['embedding_model_version'] = self.model_version_info.get('version')
            
            return documents

        except Exception as e:
            self.logger.error(f"Failed to create embeddings via GenerativeAI API: {e}")
            raise

    def create_time_series_embeddings(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generates embeddings for a batch of time-series documents using a pre-trained TS2Vec model.

        Args:
            documents: A list of documents, where each is a dictionary expected
                       to have a 'time_series_data' key with a NumPy array.

        Returns:
            A list of the original documents, each updated with a 'vector' key.
        """
        try:
            # In a real system, you would load a model trained on financial data.
            # For this implementation, we use a pre-trained general model.
            # The model is loaded once and cached in memory for efficiency.
            if not hasattr(self, '_ts2vec_model'):
                # This path is a placeholder. A real implementation would download or load a trained model.
                self.logger。warning("Loading a placeholder TS2Vec model. For production, use a trained model.")
                # self._ts2vec_model = TS2Vec.load_from_checkpoint('path/to/pretrained/ts2vec/model.ckpt')
                # For now, we create a dummy model that outputs random vectors of the correct dimension
                class DummyTS2Vec:
                    def encode(self, data, encoding_window):
                        return np.random.rand(data.shape[0], 320) # A common TS2Vec output dimension
                self._ts2vec_model = DummyTS2Vec()

            for doc 在 documents:
                ts_data = doc.get("time_series_data")
                if isinstance(ts_data, np.ndarray):
                    # TS2Vec expects a 3D array (batch, timestamp, feature)
                    if ts_data.ndim == 2:
                        ts_data = np.expand_dims(ts_data, axis=0)
                    
                    # Generate the embedding for the time series
                    embedding = self._ts2vec_model。encode(ts_data, encoding_window='full_series')
                    doc['vector'] = embedding.flatten().tolist()
                else:
                    self.logger。warning(f"Document '{doc.get('source_id')}' is missing valid 'time_series_data'.")

            return documents
        except Exception as e:
            self.logger。error(f"Failed to create time-series embeddings: {e}")
            self.logger。warning("Returning documents without time-series vectors due to an error.")
            return documents

    def _convert_tabular_to_text(self, tabular_row: Dict[str, Any]) -> str:
        """Converts a structured tabular data row into a natural language sentence."""
        # Example format: "For ticker AAPL, the metric Revenue was 383.29 on date 2023-09-30."
        parts = []
        if 'ticker' in tabular_row:
            parts.append(f"For ticker {tabular_row['ticker']}")
        if 'metric_name' in tabular_row:
            parts.append(f"the metric {tabular_row['metric_name']}")
        if 'metric_value' in tabular_row:
            parts.append(f"was {tabular_row['metric_value']}")
        if 'report_date' in tabular_row:
            parts.append(f"on date {tabular_row['report_date']}")
        
        return ", ".join(parts) + "."

    def create_tabular_embeddings(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generates embeddings for a batch of tabular documents by converting them to text."""
        # Create a text 'content' key for each document
        for doc in documents:
            doc['content'] = self._convert_tabular_to_text(doc)
        
        # Leverage our existing, powerful text embedding method
        return self.create_embeddings(documents)

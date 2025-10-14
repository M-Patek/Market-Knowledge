# ai/contradiction_detector.py
"""
Implements a service for adversarial validation and contradiction detection
within a set of evidence.
"""
import logging
import itertools
import numpy as np
from typing import List, Dict, Any, Tuple

from .embedding_client import EmbeddingClient
from ai.validation import EvidenceItem

class ContradictionDetector:
    """
    Identifies contradictory evidence items using semantic similarity and score opposition.
    """
    def __init__(self,
                 embedding_client: EmbeddingClient,
                 similarity_threshold: float = 0.85,
                 positive_threshold: float = 0.7,
                 negative_threshold: float = 0.3):
        """
        Initializes the detector.

        Args:
            embedding_client: Client to generate embeddings for evidence findings.
            similarity_threshold: Cosine similarity score above which findings are "similar".
            positive_threshold: Evidence score above which a finding is "positive".
            negative_threshold: Evidence score below which a finding is "negative".
        """
        self.logger = logging.getLogger("PhoenixProject.ContradictionDetector")
        self.embedding_client = embedding_client
        self.similarity_threshold = similarity_threshold
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold

    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Calculates the cosine similarity between two vectors."""
        return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))

    def detect(self, evidence_list: List[EvidenceItem]) -> List[Tuple[EvidenceItem, EvidenceItem]]:
        """
        Detects pairs of contradictory evidence in a given list.

        Returns:
            A list of tuples, where each tuple contains a pair of contradictory EvidenceItems.
        """
        if len(evidence_list) < 2:
            return []

        # 1. Generate embeddings for all evidence findings that have content
        docs_to_embed = [{"content": ev.finding, "_original_item": ev} for ev in evidence_list if ev.finding]
        embedded_docs = self.embedding_client.create_embeddings(docs_to_embed)

        # Filter out any that failed embedding
        valid_evidence = [doc for doc in embedded_docs if 'vector' in doc]
        if len(valid_evidence) < 2:
            return []

        contradictions = []
        
        # 2. Perform vectorized cosine similarity calculation
        embeddings = np.array([doc['vector'] for doc in valid_evidence])
        # Normalize each vector to unit length
        norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Calculate the cosine similarity matrix
        similarity_matrix = np.dot(norm_embeddings, norm_embeddings.T)

        # 3. Find pairs above the threshold and check for score opposition
        # We use np.triu_indices to only check the upper triangle of the matrix, avoiding self-comparison and duplicates
        indices_row, indices_col = np.where(similarity_matrix > self.similarity_threshold)

        for i, j in zip(indices_row, indices_col):
            if i >= j: continue # Only consider pairs where i < j

            item_i = valid_evidence[i]['_original_item']
            item_j = valid_evidence[j]['_original_item']

            is_i_positive = item_i.score >= self.positive_threshold
            is_j_positive = item_j.score >= self.positive_threshold
            is_i_negative = item_i.score <= self.negative_threshold
            is_j_negative = item_j.score <= self.negative_threshold

            if (is_i_positive and is_j_negative) or (is_i_negative and is_j_positive):
                self.logger.warning(f"CONTRADICTION DETECTED (Similarity: {similarity_matrix[i, j]:.2f}): "
                                    f"'{item_i.finding}' vs '{item_j.finding}'")
                contradictions.append((item_i, item_j))
        
        return contradictions

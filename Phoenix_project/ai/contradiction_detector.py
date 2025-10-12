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

        # 1. Generate embeddings for all evidence findings
        docs_to_embed = [{"content": ev.finding, "_original_item": ev} for ev in evidence_list]
        embedded_docs = self.embedding_client.create_embeddings(docs_to_embed)

        # Filter out any that failed embedding
        valid_evidence = [doc for doc in embedded_docs if 'vector' in doc]

        contradictions = []
        # 2. Compare every pair of evidence items
        for doc_a, doc_b in itertools.combinations(valid_evidence, 2):
            vec_a = np.array(doc_a['vector'])
            vec_b = np.array(doc_b['vector'])
            
            similarity = self._cosine_similarity(vec_a, vec_b)

            if similarity > self.similarity_threshold:
                # The findings are semantically similar, now check for score opposition
                item_a = doc_a['_original_item']
                item_b = doc_b['_original_item']

                is_a_positive = item_a.score >= self.positive_threshold
                is_b_positive = item_b.score >= self.positive_threshold
                is_a_negative = item_a.score <= self.negative_threshold
                is_b_negative = item_b.score <= self.negative_threshold

                if (is_a_positive and is_b_negative) or (is_a_negative and is_b_positive):
                    self.logger.warning(f"CONTRADICTION DETECTED (Similarity: {similarity:.2f}): "
                                        f"'{item_a.finding}' vs '{item_b.finding}'")
                    contradictions.append((item_a, item_b))
        
        return contradictions

# ai/retriever.py
"""
Implements the two-stage (recall -> re-rank) hybrid retrieval system.
This module is responsible for fetching a broad set of candidate documents
and then intelligently re-ranking them to find the most relevant, timely,
and authoritative evidence for the AI cognitive layer.
"""
import logging
import asyncio
import itertools
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta

from .vector_db_client import VectorDBClient
from .embedding_client import EmbeddingClient
from .temporal_db_client import TemporalDBClient
from .tabular_db_client import TabularDBClient

class HybridRetriever:
    """
    Orchestrates the recall and re-rank stages of evidence retrieval.
    """
    def __init__(self,
                 vector_db_client: VectorDBClient,
                 temporal_db_client: TemporalDBClient,
                 tabular_db_client: TabularDBClient,
                 embedding_client: EmbeddingClient,
                 rerank_config: Dict[str, Any]):
        """
        Initializes the retriever with clients for the various indexes.

        Args:
            vector_db_client: The client for the vector database (Pinecone).
            temporal_db_client: The client for the temporal index (Elasticsearch).
            tabular_db_client: The client for the tabular index (PostgreSQL).
            embedding_client: The client for generating query embeddings.
            rerank_config: Configuration for the re-ranking algorithm.
        """
        self.logger = logging.getLogger("PhoenixProject.HybridRetriever")
        self.vector_db_client = vector_db_client
        self.temporal_client = temporal_db_client
        self.tabular_client = tabular_db_client
        self.embedding_client = embedding_client
        
        # Re-ranking parameters
        self.rerank_config = rerank_config
        self.logger.info("HybridRetriever initialized.")

    async def _recall_from_vector(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Recall from the vector database."""
        query_doc = {"content": query}
        doc_with_vector = self.embedding_client.create_embeddings([query_doc])
        if not doc_with_vector or 'vector' not in doc_with_vector[0]:
            self.logger.error("Failed to generate query embedding for vector recall.")
            return []
        query_vector = doc_with_vector[0]['vector']
        
        results = self.vector_db_client.index.query(
            vector=query_vector, top_k=top_k, include_metadata=True
        )
        
        candidates = []
        for match in results.get('matches', []):
            candidate = match['metadata']
            candidate['vector_similarity_score'] = match.get('score', 0.0)
            candidates.append(candidate)
        return candidates

    async def _recall_from_temporal(self, entities: List[str], days_back: int = 90) -> List[Dict[str, Any]]:
        """Recall from the temporal database."""
        if not self.temporal_client.is_healthy(): return []
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        return self.temporal_client.query_by_time_and_entities(start_date, end_date, entities)

    async def _recall_from_tabular(self, ticker: str, metric: str) -> List[Dict[str, Any]]:
        """Recall from the tabular database."""
        if not self.tabular_client.is_healthy(): return []
        results = self.tabular_client.query_by_metric(ticker, metric)
        return results if results is not None else []


    async def recall(self, query: str, ticker: str, top_k: int = 50) -> List[Dict[str, Any]]:
        """
        Performs the initial recall stage by querying all indexes in parallel.
        """
        self.logger.info("Initiating parallel recall from all indexes...")
        tasks = []

        # Task for Vector DB (semantic search on the query text)
        if self.vector_db_client.is_healthy():
            tasks.append(self._recall_from_vector(query, top_k))

        # --- Simple entity/metric extraction for other indexes ---
        # In a real system, this would use a proper NLP model (e.g., NER)
        # For now, we'll assume the ticker is a key entity and look for a common metric.
        entities_in_query = [ticker]
        if "revenue" in query.lower():
            if self.tabular_client.is_healthy():
                tasks.append(self._recall_from_tabular(ticker, "Revenue"))
        
        if self.temporal_client.is_healthy():
             tasks.append(self._recall_from_temporal(entities_in_query))

        if not tasks:
            self.logger.warning("No healthy index clients available for recall.")
            return []

        # Execute all recall tasks concurrently
        results_from_all_indexes = await asyncio.gather(*tasks, return_exceptions=True)
        
        # --- Fusion Logic ---
        # Use a dictionary to merge results and ensure uniqueness by source_id
        fused_candidates = {}
        for result_set in results_from_all_indexes:
            if isinstance(result_set, list):
                for doc in result_set:
                    source_id = doc.get('source_id')
                    if not source_id: continue
                    # Merge results, prioritizing the version with more detail (e.g., from vector search)
                    if source_id not in fused_candidates or 'vector_similarity_score' in doc:
                        fused_candidates[source_id] = doc
        
        final_candidates = list(fused_candidates.values())
        self.logger.info(f"Recall stage complete. Fused {len(final_candidates)} unique candidates from all indexes.")
        return final_candidates

    def rerank(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Re-ranks a list of candidates based on a weighted formula.
        """
        scored_candidates = []
        for cand in candidates:
            score = self._calculate_rerank_score(cand)
            cand['final_rerank_score'] = score
            scored_candidates.append(cand)
        
        # Sort candidates by the final score in descending order
        return sorted(scored_candidates, key=lambda x: x['final_rerank_score'], reverse=True)

    def _calculate_rerank_score(self, candidate: Dict[str, Any]) -> float:
        """
        Calculates the final weighted score for a single candidate document.
        """
        weights = self.rerank_config.get('weights', {})
        w_similarity = weights.get('similarity', 0.6)
        w_freshness = weights.get('freshness', 0.3)
        w_source = weights.get('source', 0.1)

        # 1. Similarity Score (already normalized between 0 and 1 by Pinecone)
        similarity_score = candidate.get('vector_similarity_score', 0.0)

        # 2. Freshness Score (calculated with exponential decay)
        freshness_score = 0.0
        available_at_str = candidate.get('available_at') or candidate.get('timestamp')
        if available_at_str:
            try:
                # Handle potential timezone differences
                if isinstance(available_at_str, datetime):
                     available_at = available_at_str
                else:
                     available_at = datetime.fromisoformat(available_at_str)

                if available_at.tzinfo is None:
                    available_at = available_at.replace(tzinfo=timezone.utc)

                age_days = (datetime.now(timezone.utc) - available_at).total_seconds() / 86400
                decay_rate = self.rerank_config.get('freshness_decay_rate', 0.05)
                freshness_score = (1 - decay_rate) ** age_days
            except (ValueError, TypeError):
                pass # Use 0.0 if timestamp is invalid

        # 3. Source Weight Score (looked up from config)
        source_type = candidate.get('source_type') or candidate.get('type')
        source_weights = self.rerank_config.get('source_type_weights', {})
        source_score = source_weights.get(source_type, 0.5) # Default score for unknown sources

        # Final weighted score
        final_score = (
            w_similarity * similarity_score +
            w_freshness * freshness_score +
            w_source * source_score
        )
        return final_score

    async def retrieve(self, query: str, ticker: str, top_k_final: int = 10) -> List[Dict[str, Any]]:
        """
        Executes the full two-stage retrieval process.
        """
        self.logger.info(f"Executing two-stage retrieval for query: '{query[:50]}...'")
        # 1. Recall Stage
        candidates = await self.recall(query, ticker)
        if not candidates:
            return []
        
        # 2. Re-rank Stage
        reranked_results = self.rerank(candidates)
        
        return reranked_results[:top_k_final]

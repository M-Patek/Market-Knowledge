import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, date, timezone
import google.generativeai as genai

from .vector_db_client import VectorDBClient
from .temporal_db_client import TemporalDBClient
from .tabular_db_client import TabularDBClient
from .embedding_client import EmbeddingClient

class HybridRetriever:
    """
    Orchestrates a multi-pronged retrieval strategy across different data sources
    (vector, temporal, tabular) and fuses the results.
    """
    def __init__(self, vector_db_client: VectorDBClient, temporal_db_client: TemporalDBClient, tabular_db_client: TabularDBClient, rerank_config: Dict[str, Any]):
        self.logger = logging.getLogger("PhoenixProject.HybridRetriever")
        self.vector_client = vector_db_client
        self.temporal_client = temporal_db_client
        self.tabular_client = tabular_db_client
        self.embedding_client = EmbeddingClient()
        # RRF and Re-ranking parameters
        self.retriever_config = rerank_config # The whole retriever config section
        self.rrf_k = self.retriever_config.get('rrf_k', 60)

        # --- [NEW] Initialize the generative model for query deconstruction ---
        try:
            # Assuming the genai client is already configured elsewhere
            self.deconstructor_model = genai.GenerativeModel("gemini-1.5-flash-latest")
        except Exception as e:
            self.logger.error(f"Failed to initialize deconstructor model: {e}")
            self.deconstructor_model = None
        self.logger.info(f"HybridRetriever initialized with RRF k={self.rrf_k}.")

    async def _recall_from_vector(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Recall from the vector database."""
        return self.vector_client.query(query, top_k=top_k)

    async def _recall_from_temporal(self, ticker: str, start_date: date, end_date: date) -> List[Dict[str, Any]]:
        """Recall from the temporal database."""
        # This is a placeholder for a more sophisticated temporal query
        return self.temporal_client.query_by_date(ticker, start_date, end_date)

    async def _recall_from_tabular(self, ticker: str) -> List[Dict[str, Any]]:
        """Recall from the tabular database."""
        # This is a placeholder for a more sophisticated tabular query
        return self.tabular_client.query_by_ticker(ticker)

    async def _deconstruct_query(self, query: str) -> Dict[str, Any]:
        """
        [NEW] Uses a generative model to break down a natural language query
        into structured components for targeted retrieval.
        """
        if not self.deconstructor_model:
            self.logger.warning("Deconstructor model not available. Using simple keyword extraction.")
            return {"keywords": query.split(), "ticker": None, "date_range": None}
        
        prompt = f"""
        Deconstruct the following financial query into a structured JSON object.
        Identify keywords, any specific stock tickers (if present), and any date or time ranges.
        If a date is mentioned, provide a start and end date in YYYY-MM-DD format.
        If no ticker or date is found, the value should be null.

        QUERY: "{query}"

        JSON:
        """
        try:
            response = await self.deconstructor_model.generate_content_async(prompt)
            # Basic parsing, a real implementation would have more robust error handling
            clean_response = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_response)
        except Exception as e:
            self.logger.error(f"Failed to deconstruct query with LLM: {e}. Falling back to simple extraction.")
            # Fallback for safety
            return {"keywords": query.split(), "ticker": None, "date_range": None}

    async def recall(self, query: str, ticker: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Performs the first-stage retrieval from all available data sources in parallel
        and fuses them using Reciprocal Rank Fusion.
        """
        self.logger.info(f"Initiating recall stage for query: '{query}'")
        
        # --- [NEW] Query Deconstruction ---
        # deconstructed_query = await self._deconstruct_query(query)
        # keywords = " ".join(deconstructed_query.get("keywords", []))
        # query_ticker = deconstructed_query.get("ticker") or ticker
        # For now, we'll keep it simple as the deconstructor is not fully integrated
        keywords = query
        query_ticker = ticker
        
        # --- Parallel Recall ---
        tasks = []
        if self.vector_client:
            tasks.append(self._recall_from_vector(keywords, top_k=20))
        if self.temporal_client and query_ticker:
            # Placeholder date range for temporal query
            end_date = date.today()
            start_date = date(end_date.year - 1, end_date.month, end_date.day)
            tasks.append(self._recall_from_temporal(query_ticker, start_date, end_date))
        if self.tabular_client and query_ticker:
            tasks.append(self._recall_from_tabular(query_ticker))

        # Execute all recall tasks concurrently
        results_from_all_indexes = await asyncio.gather(*tasks, return_exceptions=True)
        
        # --- [NEW] Reciprocal Rank Fusion (RRF) Logic ---
        rrf_scores = {}
        master_doc_lookup = {}

        for result_set in results_from_all_indexes:
            if isinstance(result_set, list):
                for rank, doc in enumerate(result_set):
                    source_id = doc.get('source_id')
                    if not source_id: continue
                    
                    # Store the most complete version of the document
                    if source_id not in master_doc_lookup or 'vector_similarity_score' in doc:
                        master_doc_lookup[source_id] = doc

                    # Add to the RRF score
                    rrf_scores[source_id] = rrf_scores.get(source_id, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        # Combine the docs and their RRF scores
        fused_candidates = {}
        for source_id, score in rrf_scores.items():
            if source_id in master_doc_lookup:
                doc = master_doc_lookup[source_id]
                doc['rrf_score'] = score
                fused_candidates[source_id] = doc

        final_candidates = sorted(fused_candidates.values(), key=lambda x: x['rrf_score'], reverse=True)
        self.logger.info(f"Recall stage complete. Fused and ranked {len(final_candidates)} unique candidates using RRF.")
        return final_candidates

    def rerank(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Re-ranks a list of candidates based on a weighted formula.
        """
        # This now acts as a second-stage re-ranker on the RRF results
        weights = self.retriever_config.get('weights', {})
        w_rrf = weights.get('rrf', 0.7) # Give high weight to the RRF score
        w_freshness = weights.get('freshness', 0.2)
        w_source = weights.get('source', 0.1)
        freshness_decay_rate = self.retriever_config.get('freshness_decay_rate', 0.05)
        source_weights = self.retriever_config.get('source_type_weights', {})

        scored_candidates = []
        for cand in candidates:
            final_score = self._calculate_final_score(cand, w_rrf, w_freshness, w_source, freshness_decay_rate, source_weights)
            cand['final_score'] = final_score
            scored_candidates.append(cand)

        # Sort by the new final score
        return sorted(scored_candidates, key=lambda x: x['final_score'], reverse=True)

    def _calculate_final_score(self, candidate: Dict[str, Any], w_rrf: float, w_freshness: float, w_source: float, freshness_decay_rate: float, source_weights: Dict[str, float]) -> float:
        """
        Calculates the final weighted score for a single candidate document.
        """
        # 1. RRF Score (now the primary score from the fusion stage)
        rrf_score = candidate.get('rrf_score', 0.0)

        # 2. Freshness Score (calculated with exponential decay)
        freshness_score = 0.0
        try:
            doc_date_str = candidate.get('metadata', {}).get('document_date')
            if doc_date_str:
                doc_date = datetime.fromisoformat(doc_date_str.replace("Z", "+00:00")).astimezone(timezone.utc).date()
                days_old = (date.today() - doc_date).days
                # Exponential decay formula
                freshness_score = 1.0 * (1 - freshness_decay_rate) ** days_old
        except Exception as e:
            self.logger.warning(f"Could not parse date for freshness score calculation: {e}")

        # 3. Source Authority Score
        source_type = candidate.get('metadata', {}).get('source_type', 'Other')
        source_score = source_weights.get(source_type, 0.5) # Default score for unknown sources

        # Final weighted score
        final_score = (
            w_rrf * rrf_score +
            w_freshness * freshness_score +
            w_source * source_score
        )
        return final_score

    async def retrieve(self, query: str, ticker: Optional[str] = None, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        The main public method to perform the full retrieval and re-ranking pipeline.
        """
        self.logger.info(f"--- [Hybrid Retriever]: Full retrieval pipeline initiated for query: '{query}' ---")
        recalled_candidates = await self.recall(query, ticker)
        reranked_results = self.rerank(recalled_candidates)
        final_top_k = reranked_results[:top_k]
        self.logger.info(f"Retrieval pipeline complete. Returning top {len(final_top_k)} results.")
        return final_top_k

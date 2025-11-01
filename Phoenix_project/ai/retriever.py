import asyncio
from typing import List, Dict, Any, Optional
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

from monitor.logging import get_logger
from ai.tabular_db_client import TabularDBClient
from ai.temporal_db_client import TemporalDBClient
# FIX: Removed the try...except block for the missing 'ai.vector_db_client'
# FIX: Imported the correct VectorStore from the 'memory' module
from memory.vector_store import VectorStore
from api.gemini_pool_manager import GeminiPoolManager
from core.schemas.data_schema import QueryResult

logger = get_logger('HybridRetriever')

class HybridRetriever:
    """
    Implements the RAG (Retrieval-Augmented Generation) retrieval pipeline.
    It combines results from three different data sources:
    1. Vector Store (Semantic search for unstructured data)
    2. Temporal DB (Time-series search for events)
    3. Tabular DB (Structured search for financial metrics)
    """

    def __init__(self, 
                 config: Dict[str, Any],
                 tabular_client: TabularDBClient,
                 temporal_client: TemporalDBClient,
                 # FIX: Updated type hint to use the correct VectorStore class
                 vector_client: VectorStore,
                 gemini_pool: Optional[GeminiPoolManager]):
        """
        Initializes the HybridRetriever.
        
        Args:
            config: Configuration settings for the retriever.
            tabular_client: An initialized TabularDBClient.
            temporal_client: An initialized TemporalDBClient.
            vector_client: An initialized VectorStore client.
            gemini_pool: The shared GeminiPoolManager for LLM calls (e.g., HyDE).
        """
        self.config = config
        self.tabular_client = tabular_client
        self.temporal_client = temporal_client
        self.vector_client = vector_client
        self.gemini_pool = gemini_pool

        self.use_hyde = self.config.get('use_hyde', True)
        self.vector_namespace = self.config.get('vector_namespace', 'default')
        
        # RRF (Reciprocal Rank Fusion) constant
        self.k_rrf = self.config.get('rrf_k', 60)
        
        logger.info("HybridRetriever initialized.")
        logger.info(f"HyDE (Hypothetical Document Embeddings) enabled: {self.use_hyde}")

    async def _generate_hyde_query(self, query: str) -> str:
        """
        Generates a hypothetical document for the query (HyDE) to improve
        semantic retrieval quality.
        """
        if not self.gemini_pool:
            logger.warning("HyDE enabled but no Gemini pool available. Skipping.")
            return query
            
        prompt = f"""
        Please generate a short, hypothetical document that provides a relevant answer 
        to the following query. The document should be neutral, factual, and dense 
        with information related to the query.
        
        Query: "{query}"
        
        Hypothetical Document:
        """
        
        try:
            model = self.config.get('hyde_model', 'gemini-1.5-flash-latest')
            response_text = await self.gemini_pool.generate_content(prompt, model_name=model)
            logger.debug(f"Generated HyDE document for query '{query}': {response_text[:100]}...")
            # Combine original query with HyDE doc for embedding
            return f"{query}\n\n{response_text}"
        except Exception as e:
            logger.error(f"Failed to generate HyDE query: {e}", exc_info=True)
            # Fallback to the original query
            return query

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _query_vector_store(self, query: str, top_k: int) -> List[QueryResult]:
        """Wrapper for vector store query with retry logic."""
        try:
            hyde_query = query
            if self.use_hyde:
                hyde_query = await self._generate_hyde_query(query)
                
            results = await self.vector_client.query(
                query=hyde_query,
                top_k=top_k,
                namespace=self.vector_namespace
                # TODO: Add metadata filtering if needed
                # filter={"source": "sec_filings"}
            )
            # Convert to standard QueryResult objects
            return [
                QueryResult(
                    id=res.get('id'),
                    text=res.get('text', ''),
                    score=res.get('score', 0.0),
                    source=res.get('metadata', {}).get('source', 'vector_store'),
                    type='vector'
                ) for res in results
            ]
        except Exception as e:
            logger.error(f"Vector store query failed: {e}", exc_info=True)
            raise  # Re-raise to trigger tenacity retry

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _query_temporal_store(self, query: str, top_k: int, time_window_days: int) -> List[QueryResult]:
        """Wrapper for temporal store query with retry logic."""
        try:
            results = await self.temporal_client.search_events(
                keywords=query.split(), # Simple keyword search
                top_k=top_k,
                days=time_window_days
            )
            return [
                QueryResult(
                    id=res.get('event_id'),
                    text=res.get('content', ''),
                    score=res.get('score', 0.0), # Assuming temporal client provides a score
                    source=res.get('source', 'temporal_store'),
                    type='temporal'
                ) for res in results
            ]
        except Exception as e:
            logger.error(f"Temporal store query failed: {e}", exc_info=True)
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _query_tabular_store(self, query: str, top_k: int) -> List[QueryResult]:
        """Wrapper for tabular store query with retry logic."""
        # This is a simplified example. A real implementation would parse 'query'
        # to identify symbols, metrics, and quarters.
        # For now, we'll assume the query is just a symbol.
        symbol = query.upper() # e.g., "AAPL"
        try:
            results = await self.tabular_client.get_financials(symbol, limit=top_k)
            return [
                QueryResult(
                    id=f"{res.get('symbol')}_{res.get('quarter')}",
                    text=f"Financials for {res.get('symbol')} (Q{res.get('quarter')} {res.get('year')}): "
                         f"Revenue: {res.get('revenue')}, EPS: {res.get('eps')}",
                    score=1.0, # Tabular data is exact match, give high score
                    source='tabular_store',
                    type='tabular'
                ) for res in results
            ]
        except Exception as e:
            logger.error(f"Tabular store query failed for symbol '{symbol}': {e}", exc_info=True)
            raise

    def _fuse_results_rrf(self, results_lists: List[List[QueryResult]]) -> List[QueryResult]:
        """
        Fuses multiple ranked lists of results using Reciprocal Rank Fusion (RRF).
        """
        fused_scores = {}
        doc_content = {}
        
        for results in results_lists:
            if not results:
                continue
            for rank, res in enumerate(results):
                doc_id = res.id
                if not doc_id:
                    continue
                    
                score = 1.0 / (self.k_rrf + rank + 1)
                
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0.0
                    doc_content[doc_id] = res # Store the full QueryResult object
                
                fused_scores[doc_id] += score
        
        if not fused_scores:
            return []

        # Sort by fused score
        sorted_docs = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        
        # Reconstruct final list of QueryResult objects
        final_results = []
        for doc_id, score in sorted_docs:
            final_doc = doc_content[doc_id]
            final_doc.score = score # Overwrite with fused score
            final_results.append(final_doc)
            
        return final_results

    async def retrieve(self, 
                       query: str, 
                       top_k: int = 10, 
                       time_window_days: int = 30) -> List[QueryResult]:
        """
        Executes the full hybrid retrieval pipeline.
        
        1. Asynchronously query all three data sources.
        2. Fuse the results using Reciprocal Rank Fusion (RRF).
        3. Return the final, reranked list of documents.
        """
        logger.info(f"Starting hybrid retrieval for query: '{query}'")
        
        try:
            # 1. Dispatch all queries concurrently
            tasks = [
                self._query_vector_store(query, top_k),
                self._query_temporal_store(query, top_k, time_window_days),
                self._query_tabular_store(query, top_k)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle potential failures
            vector_results = results[0] if not isinstance(results[0], Exception) else []
            temporal_results = results[1] if not isinstance(results[1], Exception) else []
            tabular_results = results[2] if not isinstance(results[2], Exception) else []
            
            if isinstance(results[0], Exception):
                 logger.error(f"Vector query failed after retries: {results[0]}")
            if isinstance(results[1], Exception):
                 logger.error(f"Temporal query failed after retries: {results[1]}")
            if isinstance(results[2], Exception):
                 logger.error(f"Tabular query failed after retries: {results[2]}")

            # 2. Fuse the results
            all_results = [vector_results, temporal_results, tabular_results]
            fused_list = self._fuse_results_rrf(all_results)
            
            # 3. Get the final top_k results
            final_top_k = fused_list[:top_k]
            
            logger.info(f"Hybrid retrieval complete. Found {len(final_top_k)} fused results (from "
                        f"{len(vector_results)} vector, {len(temporal_results)} temporal, "
                        f"{len(tabular_results)} tabular).")
            
            return final_top_k
            
        except Exception as e:
            logger.error(f"An unexpected error occurred during hybrid retrieval: {e}", exc_info=True)
            return []

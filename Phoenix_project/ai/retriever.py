import asyncio
from typing import Dict, Any, List, Optional, Tuple

from ..data_manager import DataManager
from ..memory.vector_store import VectorStore
from .temporal_db_client import TemporalDBClient
from .tabular_db_client import TabularDBClient
from .embedding_client import EmbeddingClient
from ..monitor.logging import get_logger
from ..core.schemas.data_schema import MarketEvent, TickerData

logger = get_logger(__name__)

class Retriever:
    """
    Performs hybrid RAG (Retrieval-Augmented Generation) by querying multiple
    data sources (Vector, Temporal, Tabular) in parallel to build a
    comprehensive context bundle.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        data_manager: DataManager,
        vector_store: VectorStore,
        temporal_db: TemporalDBClient,
        tabular_db: TabularDBClient,
        embedding_client: EmbeddingClient
    ):
        """
        Initializes the Retriever with clients for all required data sources.
        """
        self.config = config.get('retriever', {})
        self.data_manager = data_manager
        self.vector_store = vector_store
        self.temporal_db = temporal_db
        self.tabular_db = tabular_db
        self.embedding_client = embedding_client
        
        # Get retrieval parameters from config
        self.vector_top_k = self.config.get('vector_top_k', 5)
        self.temporal_window_days = self.config.get('temporal_window_days', 7)
        
        logger.info("Retriever initialized.")

    async def retrieve_hybrid_context(
        self, 
        event: MarketEvent, 
        market_context: Optional[List[TickerData]] = None
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
        """
        Main entry point for hybrid retrieval. Executes all retrieval tasks concurrently.

        Args:
            event: The triggering MarketEvent.
            market_context: Optional list of recent TickerData.

        Returns:
            A tuple containing:
            1. context_bundle (Dict): The consolidated results, e.g.,
               {"vector_context": [...], "temporal_context": [...], "structured_context": [...]}
            2. metadata (Dict): Auditing metadata about the retrieval process.
        """
        logger.debug(f"Starting hybrid retrieval for event: {event.event_id}")
        
        # Create the text to be used for semantic search
        query_text = f"Headline: {event.headline}\nSummary: {event.summary}"
        
        # Create tasks for each retrieval source
        tasks = {
            "vector": asyncio.create_task(self._retrieve_vector_context(query_text, event)),
            "temporal": asyncio.create_task(self._retrieve_temporal_context(event)),
            "structured": asyncio.create_task(self._retrieve_structured_context(event)),
            "market": asyncio.create_task(self._retrieve_market_context(event, market_context))
        }
        
        # Wait for all tasks to complete
        await asyncio.wait(tasks.values())

        # Consolidate results
        context_bundle = {}
        metadata = {"tasks": {}}

        for name, task in tasks.items():
            try:
                result, meta = task.result()
                context_bundle[f"{name}_context"] = result
                metadata['tasks'][name] = {"status": "success", **meta}
            except Exception as e:
                logger.error(f"Retrieval task '{name}' failed for event {event.event_id}: {e}", exc_info=True)
                context_bundle[f"{name}_context"] = []
                metadata['tasks'][name] = {"status": "error", "message": str(e)}

        logger.info(f"Hybrid retrieval complete for event: {event.event_id}")
        return context_bundle, metadata

    async def _retrieve_vector_context(self, query_text: str, event: MarketEvent) -> Tuple[List[Dict], Dict]:
        """Retrieves semantic context from the VectorStore (e.g., Pinecone)."""
        start_time = asyncio.get_event_loop().time()
        
        # Get embedding for the query text
        query_embedding = await self.embedding_client.get_embedding(query_text, cache_key=f"query_{event.event_id}")
        if not query_embedding:
            logger.warning(f"Could not generate query embedding for event: {event.event_id}")
            return [], {"duration_ms": 0, "hits": 0, "error": "Embedding generation failed"}

        # Search vector store
        # TODO: Add metadata filters (e.g., date range, source)
        results = await self.vector_store.search(
            query_embedding=query_embedding,
            top_k=self.vector_top_k
        )
        
        duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        metadata = {"duration_ms": duration_ms, "hits": len(results), "query_snippet": query_text[:100]}
        
        return results, metadata

    async def _retrieve_temporal_context(self, event: MarketEvent) -> Tuple[List[Dict], Dict]:
        """Retrieves time-series context from the TemporalDB (e.g., Elasticsearch)."""
        start_time = asyncio.get_event_loop().time()
        
        end_time = event.timestamp
        start_time_filter = end_time - pd.Timedelta(days=self.temporal_window_days)
        
        results = await self.temporal_db.search_events(
            symbols=event.symbols,
            start_time=start_time_filter,
            end_time=end_time
        )
        
        # Filter out the triggering event itself
        results = [r for r in results if r.get('event_id') != event.event_id]
        
        duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        metadata = {"duration_ms": duration_ms, "hits": len(results), "window_days": self.temporal_window_days}
        
        return results, metadata

    async def _retrieve_structured_context(self, event: MarketEvent) -> Tuple[List[Dict], Dict]:
        """Retrieves structured financial data from the TabularDB (e.g., PostgreSQL)."""
        start_time = asyncio.get_event_loop().time()
        
        # Example: Get latest financial metrics for the event's symbols
        all_metrics = []
        for symbol in event.symbols:
            metrics = await self.tabular_db.get_latest_financials(symbol)
            if metrics:
                all_metrics.append(metrics)
                
        duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        metadata = {"duration_ms": duration_ms, "symbols_queried": len(event.symbols), "hits": len(all_metrics)}
        
        return all_metrics, metadata

    async def _retrieve_market_context(self, event: MarketEvent, market_context: Optional[List[TickerData]]) -> Tuple[List[Dict], Dict]:
        """Packages the already-provided market context."""
        start_time = asyncio.get_event_loop().time()
        
        if not market_context:
            # If not provided, fetch it (less efficient)
            # This is a fallback
            logger.warning(f"Market context not provided for event {event.event_id}. Fetching...")
            # This logic should be more robust, getting data for all symbols
            # For now, just return empty
            results_list = []
        else:
            # Convert Pydantic models to dicts for the context bundle
            results_list = [ticker.model_dump() for ticker in market_context]
            
        duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        metadata = {"duration_ms": duration_ms, "series_count": len(results_list)}
        
        return results_list, metadata

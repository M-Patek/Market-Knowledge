from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio

from ..core.schemas.data_schema import QueryResult
from ..memory.vector_store import VectorStore
from ..ai.temporal_db_client import TemporalDBClient
# 修复：从 'import ai.tabular_db_client' 改为导入
from ..ai.tabular_db_client import TabularDBClient # 假设类名为 TabularDBClient
from ..monitor.logging import get_logger

logger = get_logger(__name__)

class Retriever:
    """
    Hybrid Retriever (RAG)
    Combines results from Vector, Temporal, and Tabular data stores.
    """

    def __init__(self, 
                 vector_store: VectorStore, 
                 temporal_db: TemporalDBClient, 
                 tabular_db: TabularDBClient,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the retriever with clients to all data stores.
        """
        self.vector_store = vector_store
        self.temporal_db = temporal_db
        self.tabular_db = tabular_db
        self.config = config or {}
        self.top_k_vector = self.config.get('top_k_vector', 5)
        self.top_k_temporal = self.config.get('top_k_temporal', 5)
        
        logger.info("Retriever initialized with Vector, Temporal, and Tabular stores.")

    async def retrieve(self, 
                       query: str, 
                       start_date: datetime, 
                       end_date: datetime,
                       **kwargs) -> List[QueryResult]:
        """
        Asynchronously retrieves and combines context from all data stores.
        """
        logger.debug(f"Retrieving context for query: '{query}' from {start_date} to {end_date}")
        
        # 1. Create tasks for all retrievers
        vector_task = self.vector_store.search(
            query=query, 
            top_k=self.top_k_vector,
            start_date=start_date,
            end_date=end_date
        )
        
        temporal_task = self.temporal_db.query_events(
            query_text=query,
            start_time=start_date,
            end_time=end_date,
            limit=self.top_k_temporal
        )
        
        # Tabular retrieval might need query parsing first
        # For simplicity, we assume it can also take the raw query
        tabular_task = self.tabular_db.search_tables(
            query=query,
            start_date=start_date,
            end_date=end_date
        )

        # 2. Run tasks in parallel
        try:
            results = await asyncio.gather(
                vector_task, 
                temporal_task, 
                tabular_task
            )
            
            vector_results, temporal_results, tabular_results = results
            
            # 3. Combine and format all results into QueryResult schema
            all_results: List[QueryResult] = []
            
            if vector_results:
                all_results.extend([
                    QueryResult(
                        source='vector', 
                        content=r.get('text', ''), 
                        metadata=r.get('metadata', {}), 
                        score=r.get('score', 0)
                    ) for r in vector_results
                ])
                
            if temporal_results:
                all_results.extend([
                    QueryResult(
                        source='temporal', 
                        content=r.get('description', ''), 
                        metadata=r, 
                        score=r.get('relevance_score', 0) # 假设得分字段
                    ) for r in temporal_results
                ])

            if tabular_results:
                all_results.extend([
                    QueryResult(
                        source='tabular', 
                        content=str(r.get('data_snippet', '')), 
                        metadata=r,
                        score=r.get('relevance_score', 0) # 假设得分字段
                    ) for r in tabular_results
                ])

            logger.info(f"Retrieved {len(all_results)} total results from 3 sources.")
            
            # 4. (Optional) Re-rank or filter results
            # For now, just return combined list
            
            return all_results

        except Exception as e:
            logger.error(f"Error during retrieval: {e}", exc_info=True)
            return []

    def _rerank_results(self, results: List[QueryResult]) -> List[QueryResult]:
        """
        (Placeholder) Re-ranks combined results based on relevance, source, time.
        """
        # Simple sorting by score (descending) as a default
        return sorted(results, key=lambda r: r.score or 0.0, reverse=True)


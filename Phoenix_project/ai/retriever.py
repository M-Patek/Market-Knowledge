"""
Multi-source Retriever for the RAG system.

This module is responsible for fetching context (evidence) from various
data stores in parallel to provide a comprehensive knowledge base for
the AI agents.
"""
import logging
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd

# 修复：添加 TickerData 并使用正确的相对导入
from ..core.schemas.data_schema import MarketEvent, TickerData
from ..memory.vector_store import VectorStore
from .temporal_db_client import TemporalDBClient
from .tabular_db_client import TabularDBClient
from ..config.loader import ConfigLoader

logger = logging.getLogger(__name__)

class Retriever:
    """
    Orchestrates parallel data retrieval from vector, temporal, and tabular stores.
    """

    def __init__(self, vector_store: VectorStore, 
                 temporal_db: TemporalDBClient, 
                 tabular_db: TabularDBClient):
        
        self.vector_store = vector_store
        self.temporal_db = temporal_db
        self.tabular_db = tabular_db
        
        if not all([vector_store, temporal_db, tabular_db]):
            raise ValueError("All data stores (Vector, Temporal, Tabular) must be provided.")
            
        logger.info("Retriever initialized with Vector, Temporal, and Tabular stores.")

    async def retrieve_evidence(self, 
                                query: str, 
                                symbols: List[str], 
                                end_time: datetime, 
                                top_k_vector: int = 10,
                                window_days: int = 7) -> Dict[str, Any]:
        """
        Gathers all necessary evidence for a given query and context.

        Args:
            query: The main query or headline to search for.
            symbols: List of symbols involved.
            end_time: The reference "now" timestamp for retrieval.
            top_k_vector: Number of vector documents to retrieve.
            window_days: Number of past days to look back for temporal data.

        Returns:
            A dictionary containing all retrieved evidence, structured by source.
        """
        start_time = end_time - timedelta(days=window_days)
        
        # Create a list of tasks to run concurrently
        tasks = [
            self.vector_store.search(query, top_k_vector, filter_dict={"symbols": symbols}),
            self.temporal_db.search_events(symbols, start_time, end_time),
            self.tabular_db.get_financials(symbols, end_time),
            self.temporal_db.get_market_data(symbols, start_time, end_time) # For recent price action
        ]
        
        try:
            # Run tasks in parallel
            results = await asyncio.gather(*tasks)
            
            vector_results, temporal_events, tabular_data, market_data = results
            
            # Structure the final output
            evidence = {
                "retrieval_timestamp": datetime.now().isoformat(),
                "query": query,
                "context_window": {"start": start_time.isoformat(), "end": end_time.isoformat()},
                "vector_context": [doc.to_dict() for doc in vector_results] if vector_results else [],
                "temporal_context": [event.dict() for event in temporal_events] if temporal_events else [],
                "tabular_context": tabular_data if tabular_data else {},
                "market_context": [tick.dict() for tick in market_data] if market_data else [],
            }
            
            logger.info(f"Retrieved evidence for query '{query}': "
                        f"{len(evidence['vector_context'])} vector docs, "
                        f"{len(evidence['temporal_context'])} temporal events, "
                        f"{len(evidence['tabular_context'])} tabular records, "
                        f"{len(evidence['market_context'])} market data points.")
            
            return evidence
            
        except Exception as e:
            logger.error(f"Error during parallel evidence retrieval: {e}", exc_info=True)
            return {"error": str(e), "vector_context": [], "temporal_context": [], "tabular_context": {}, "market_context": []}

# Example usage (simulated)
if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)
    
    # --- Mock Data Stores ---
    class MockVectorStore:
        async def search(self, query, top_k, filter_dict=None):
            logger.info(f"[MockVector] Searching for '{query}' with filter {filter_dict}")
            # Mock a document
            class MockDoc:
                def to_dict(self):
                    return {"id": "vec_123", "content": "Mock vector content about " + query, "score": 0.9}
            return [MockDoc()]

    class MockTemporalDB:
        async def search_events(self, symbols, start, end):
            logger.info(f"[MockTemporal] Searching for events for {symbols} from {start} to {end}")
            return [MarketEvent(event_id="evt_456", timestamp=datetime.now(), source="MockSource",
                                headline="Mock event headline", content="...", symbols=symbols)]
        
        async def get_market_data(self, symbols, start, end):
            logger.info(f"[MockTemporal] Searching for market data for {symbols} from {start} to {end}")
            return [TickerData(symbol=s, timestamp=datetime.now(), open=1, high=2, low=1, close=2, volume=1000) for s in symbols]

    class MockTabularDB:
        async def get_financials(self, symbols, end_time):
            logger.info(f"[MockTabular] Getting financials for {symbols}")
            return {s: {"latest_eps": 1.25, "pe_ratio": 20.5} for s in symbols}

    # --- Run Example ---
    async def main():
        retriever = Retriever(
            vector_store=MockVectorStore(),
            temporal_db=MockTemporalDB(),
            tabular_db=MockTabularDB()
        )
        
        query_headline = "Major tech breakthrough announced by AAPL"
        query_symbols = ["AAPL", "MSFT"]
        query_time = datetime.now()
        
        evidence = await retriever.retrieve_evidence(
            query=query_headline,
            symbols=query_symbols,
            end_time=query_time
        )
        
        import json
        print("\n--- Retrieved Evidence ---")
        print(json.dumps(evidence, indent=2, default=str))
        print("--------------------------")
        
        # Test error handling
        class FailingVectorStore:
            async def search(self, query, top_k, filter_dict=None):
                raise Exception("Vector store connection failed!")
        
        retriever_fail = Retriever(
            vector_store=FailingVectorStore(),
            temporal_db=MockTemporalDB(),
            tabular_db=MockTabularDB()
        )
        evidence_fail = await retriever_fail.retrieve_evidence(query_headline, query_symbols, query_time)
        print("\n--- Failed Retrieval ---")
        print(json.dumps(evidence_fail, indent=2, default=str))
        print("------------------------")

    asyncio.run(main())

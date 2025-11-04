from typing import List, Dict, Any, Optional
from Phoenix_project.ai.embedding_client import EmbeddingClient
from Phoenix_project.memory.vector_store import VectorStore
from Phoenix_project.memory.cot_database import CoTDatabase
from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

class Retriever:
    """
    Retrieves relevant information from various memory stores
    (VectorStore, CoTDatabase) to build context for the AI agents.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        cot_database: CoTDatabase,
        embedding_client: EmbeddingClient
    ):
        self.vector_store = vector_store
        self.cot_database = cot_database
        self.embedding_client = embedding_client
        logger.info("Retriever initialized.")

    async def retrieve_relevant_context(
        self,
        query: str,
        top_k_vector: int = 5,
        top_k_cot: int = 3,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[Any]]:
        """
        Retrieves context from all available stores based on a query.
        
        Args:
            query (str): The search query (e.g., summary of recent events,
                         or a specific question).
            top_k_vector (int): Number of results from VectorStore.
            top_k_cot (int): Number of results from CoTDatabase.
            metadata_filter (Optional[Dict[str, Any]]): Filter for vector search.

        Returns:
            A dictionary containing the retrieved "vector_chunks" and "cot_traces".
        """
        logger.info(f"Retrieving context for query: {query[:50]}...")
        
        # 1. Generate embedding for the query
        try:
            query_embedding = await self.embedding_client.get_embedding(query)
            if not query_embedding:
                logger.error("Failed to generate query embedding. Cannot perform vector search.")
                vector_chunks = []
            else:
                # 2. Query VectorStore
                vector_chunks = await self.vector_store.search(
                    query_embedding=query_embedding,
                    top_k=top_k_vector,
                    metadata_filter=metadata_filter
                )
                logger.info(f"Retrieved {len(vector_chunks)} chunks from VectorStore.")
                
        except Exception as e:
            logger.error(f"Error during vector retrieval: {e}", exc_info=True)
            vector_chunks = []

        # 3. Query CoTDatabase (e.g., by keyword matching on the query)
        try:
            # CoTDatabase search might be simpler (e.g., keyword or tag based)
            # This is a placeholder for a real search implementation
            cot_traces = await self.cot_database.search_traces(
                keywords=query.split(), # Simple keyword search
                limit=top_k_cot
            )
            logger.info(f"Retrieved {len(cot_traces)} traces from CoTDatabase.")
        except Exception as e:
            logger.error(f"Error during CoT retrieval: {e}", exc_info=True)
            cot_traces = []

        return {
            "vector_chunks": vector_chunks,
            "cot_traces": cot_traces
        }

    async def retrieve_for_context_window(
        self,
        base_query: str,
        max_tokens: int,
        # ... other params
    ) -> str:
        """
        A more advanced retrieval method that fetches context and formats it
        to fit within a specific token limit for an LLM prompt.
        """
        # This is a complex task ("RAG pipeline")
        # 1. Retrieve more than needed
        retrieved_data = await self.retrieve_relevant_context(
            base_query, top_k_vector=10, top_k_cot=5
        )
        
        # 2. Re-rank results (e.g., using a cross-encoder, not implemented)
        
        # 3. Stuff into context window
        formatted_context = "--- Relevant Knowledge ---\n\n"
        
        # Add CoT traces first (often high-value)
        for trace in retrieved_data["cot_traces"]:
            trace_text = f"Previous Reasoning ({trace['timestamp']}):\n{trace['reasoning']}\nDecision: {trace['decision']}\n\n"
            if len(formatted_context) + len(trace_text) > max_tokens:
                break
            formatted_context += trace_text
            
        # Add vector chunks
        for chunk in retrieved_data["vector_chunks"]:
            chunk_text = f"Retrieved Document (Source: {chunk.get('source', 'N/A')}):\n{chunk.get('text', '')}\n\n"
            if len(formatted_context) + len(chunk_text) > max_tokens:
                break
            formatted_context += chunk_text

        if len(formatted_context) > max_tokens:
             formatted_context = formatted_context[:max_tokens] + "... [Truncated]"
             
        logger.info(f"Assembled context window of {len(formatted_context)} chars.")
        return formatted_context

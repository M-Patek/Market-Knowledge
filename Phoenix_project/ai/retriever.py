"""
Advanced Retrieval module for the Phoenix project.

Handles hybrid search (vector, keyword, graph) and reranking.
Optimized for Fault Tolerance (Task 1), Security (Task 2), and Resource Management (Task 5).
"""

import logging
import asyncio
from asyncio.exceptions import TimeoutError
from typing import Any, List, Dict, Optional

from sentence_transformers import CrossEncoder  # type: ignore
from celery import current_app # [Task 5B] For sending tasks by name
from celery.result import AsyncResult # [Task 5B] For typing

from ..core.schemas.evidence_schema import (
    Evidence,
    QueryResult,
    GraphQueryResult,
    DataSource,
)
from ..memory.vector_store import VectorStore
from ..core.exceptions import QuerySafetyError, CognitiveError
from ..ai.graph_db_client import GraphDBClient
from ..ai.gnn_inferencer import GNNInferencer
from ..ai.tabular_db_client import TabularDBClient
from ..ai.temporal_db_client import TemporalDBClient
from ..ai.ensemble_client import EnsembleClient
from ..ai.prompt_manager import PromptManager
from ..ai.prompt_renderer import PromptRenderer


# 配置日志
logger = logging.getLogger(__name__)


class Retriever:
    """
    The Retriever class is responsible for fetching and reranking relevant
    information from various data sources (VectorDB, GraphDB, etc.)
    based on a query.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        graph_db: GraphDBClient,
        config: Dict[str, Any],
        prompt_manager: PromptManager,
        prompt_renderer: PromptRenderer,
        gnn_inferencer: GNNInferencer,
        ensemble_client: EnsembleClient,
        temporal_db: Optional[TemporalDBClient] = None,
        tabular_db: Optional[TabularDBClient] = None,
        search_client: Optional[Any] = None
    ):
        self.vector_store = vector_store
        self.graph_db = graph_db
        self.gnn_inferencer = gnn_inferencer
        
        self.ensemble_client = ensemble_client
        self.llm_client = ensemble_client.api_gateway 
        
        self.temporal_db = temporal_db
        self.tabular_db = tabular_db
        self.search_client = search_client

        self.prompt_manager = prompt_manager
        self.prompt_renderer = prompt_renderer
        
        self.config = config 

        default_l1_config = self.config.get("l1_retriever", {})

        self.rerank_model_name = default_l1_config.get(
            "rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self.top_k_vector = default_l1_config.get("vector_top_k", 10)
        self.top_k_graph = default_l1_config.get("top_k_graph", 5)
        self.top_k_gnn = default_l1_config.get("top_k_gnn", 5)
        self.rerank_threshold = default_l1_config.get("rerank_threshold", 0.1)
        self.min_successful_sources = default_l1_config.get("min_successful_sources", 2)

        self._initialize_reranker()

    def _initialize_reranker(self):
        """
        Initializes the CrossEncoder model for reranking.
        Includes a fallback mechanism.
        """
        try:
            self.reranker = CrossEncoder(self.rerank_model_name)
            logger.info(f"Successfully loaded reranker model: {self.rerank_model_name}")
        except ImportError as e:
            logger.error(
                f"[Reranker Load Failure] CrossEncoder or dependencies not found: {e}. "
                f"Falling back to simple reranker.",
                exc_info=True
            )
            self.reranker = None
        except Exception as e:
            logger.error(
                f"[Reranker Load Failure] Failed to load CrossEncoder model '{self.rerank_model_name}'. "
                f"Error: {e}. Falling back to simple reranker.",
                exc_info=True
            )
            self.reranker = None

    def _fallback_reranker(
        self, query: str, documents: List[Evidence]
    ) -> List[Evidence]:
        """
        A simple fallback reranker if the CrossEncoder model fails to load.
        """
        logger.warning("Using fallback reranker (no reranking).")
        return sorted(documents, key=lambda x: x.score, reverse=True)[
            : self.top_k_vector
        ]

    def _rerank_documents(
        self, query: str, documents: List[Evidence]
    ) -> List[Evidence]:
        """
        Reranks a list of documents against a query using the CrossEncoder.
        """
        if not documents:
            return []

        if self.reranker is None:
            return self._fallback_reranker(query, documents)

        try:
            pairs: List[List[str]] = [[query, doc.content] for doc in documents]
            if not pairs:
                return []

            scores = self.reranker.predict(pairs)

            reranked_docs: List[Evidence] = []
            for i, doc in enumerate(documents):
                new_score = float(scores[i])
                if new_score >= self.rerank_threshold:
                    doc.score = new_score
                    doc.source_type = DataSource.RERANKED_VECTOR
                    reranked_docs.append(doc)

            reranked_docs.sort(key=lambda x: x.score, reverse=True)

            logger.info(
                f"Reranked {len(documents)} docs down to {len(reranked_docs)}."
            )
            return reranked_docs

        except Exception as e:
            logger.error(
                f"[Rerank Failure] Error during document reranking: {e}",
                exc_info=True
            )
            return sorted(documents, key=lambda x: x.score, reverse=True)[
                : self.top_k_vector
            ]

    async def search_vector_db(self, query: str) -> List[Evidence]:
        """
        Performs a vector search in the VectorStore.
        """
        try:
            logger.debug(f"Performing vector search for: {query}")
            results = await self.vector_store.search(
                query_text=query, k=self.top_k_vector
            )
            logger.info(f"Vector search found {len(results)} results.")
            return results
        except (ConnectionError, TimeoutError) as e:
            logger.error(
                f"[RAG Retrieval Failure] Connection/Timeout error during vector search: {e}",
                exc_info=True
            )
            return []
        except Exception as e:
            logger.error(
                f"[RAG Retrieval Failure] Unexpected error during vector search: {e}",
                exc_info=True
            )
            return []

    def _generate_cypher_prompt(self, query: str, schema: str) -> str:
        """
        [Task 4] 重构：使用 PromptRenderer 生成 Cypher 提示。
        """
        logger.debug("Rendering text_to_cypher prompt...")
        
        try:
            context = {
                "schema": schema,
                "query": query
            }
            
            rendered_data = self.prompt_renderer.render("text_to_cypher", context)
            
            prompt_str = rendered_data.get("full_prompt_template")
            if not prompt_str:
                 raise ValueError("'full_prompt_template' key not found in rendered prompt.")
            
            return prompt_str

        except Exception as e:
            logger.error(
                f"[Prompt Render Failure] Failed to render text_to_cypher prompt: {e}. Using fallback.",
                exc_info=True
            )
            return f"""
            You are an expert Cypher query generator. Your task is to convert a
            natural language question into a Cypher query based on the provided
            Neo4j graph schema.

            Schema:
            {schema}

            Rules:
            1. Only generate the Cypher query. No preamble, no explanation.
            2. The query must be syntactically correct and relevant to the schema.
            3. Use relationship directions correctly.
            4. Use `LIMIT` clauses to keep results manageable (e.g., LIMIT 10).
            5. Focus on finding nodes and relationships relevant to the question.
            6. Return nodes or paths that answer the question.

            Question:
            "{query}"

            Cypher Query:
            """

    async def _generate_text_to_cypher(self, query: str) -> Optional[str]:
        """
        OPTIMIZED: Uses an LLM to generate a Cypher query from text.
        """
        try:
            schema = await self.graph_db.get_schema()
            if not schema:
                logger.warning("Could not retrieve graph schema for text-to-Cypher.")
                schema = "No schema available."

            prompt = self._generate_cypher_prompt(query, schema)
            
            response_text = await self.llm_client.generate_text(prompt)

            if not response_text:
                logger.warning("LLM failed to generate Cypher query.")
                return None

            cypher_query = response_text.strip().replace("```cypher", "").replace("```", "").strip()

            if not cypher_query.upper().startswith("MATCH"):
                logger.warning(f"LLM generated invalid Cypher: {cypher_query}")
                return None
            
            logger.info(f"Generated Cypher query: {cypher_query}")
            return cypher_query

        except (ConnectionError, TimeoutError) as e:
            logger.error(
                f"[RAG Retrieval Failure] Connection/Timeout error in _generate_text_to_cypher (LLM or DB call): {e}",
                exc_info=True
            )
            return None
        except Exception as e:
            logger.error(
                f"[RAG Retrieval Failure] Unexpected error in _generate_text_to_cypher: {e}",
                exc_info=True
            )
            return None


    async def search_knowledge_graph(self, query: str) -> List[GraphQueryResult]:
        """
        OPTIMIZED: Performs a search in the Knowledge Graph.
        Includes security validation (Task 2) and robust error handling (Task 1).
        """
        logger.debug(f"Performing knowledge graph search for: {query}")
        
        cypher_query = await self._generate_text_to_cypher(query)

        if not cypher_query:
            logger.warning("Text-to-Cypher failed. Falling back to keyword graph search.")
            cypher_query = f"""
            CALL db.index.fulltext.queryNodes("fulltext_nodes", $query) YIELD node, score
            RETURN node, score
            LIMIT {self.top_k_graph}
            """
            params = {"query": query}
        else:
            params = {} 

        try:
            GraphDBClient.validate_cypher_query(cypher_query)
        except QuerySafetyError as e:
            logger.error(
                f"Cypher validation failed for query: {cypher_query}. Error: {e}",
                exc_info=True
            )
            raise e
        
        try:
            results = await self.graph_db.execute_query(cypher_query, params)
            
            graph_results = []
            if results:
                for record in results:
                    node_data = record.get("node", {})
                    score = record.get("score", 0.9) 
                    
                    graph_results.append(
                        GraphQueryResult(
                            node_id=node_data.get("id", "unknown"),
                            node_type=next(iter(node_data.labels), "Unknown"),
                            content=str(node_data.properties),
                            score=float(score),
                            source_type=DataSource.GRAPH,
                        )
                    )
            
            logger.info(f"Graph search found {len(graph_results)} results.")
            return graph_results

        except (ConnectionError, TimeoutError) as e:
            logger.error(
                f"[RAG Retrieval Failure] Connection/Timeout error during graph search with query '{cypher_query}': {e}",
                exc_info=True
            )
            return []
        except Exception as e:
            logger.error(
                f"[RAG Retrieval Failure] Unexpected error during graph search with query '{cypher_query}': {e}",
                exc_info=True
            )
            return []

    async def _query_graph_with_gnn(self, query_text: str, k: int) -> List[Evidence]:
        """
        Performs a GNN-enhanced query on the knowledge graph.
        [Task 5B] Optimized to offload heavy inference to a dedicated Celery queue.
        """
        try:
            logger.debug(f"Performing GNN-enhanced graph query for: {query_text}")

            cypher_query = """
            MATCH (n)
            WHERE n.name IS NOT NULL
            RETURN {
                nodes: collect(DISTINCT n),
                edges: [] 
            } AS subgraph_data
            LIMIT 1
            """
            params = {"query": query_text, "k_seeds": k}
            
            query_results = await self.graph_db.execute_query(cypher_query, params)
            
            if not query_results or 'subgraph_data' not in query_results[0]:
                logger.warning("GNN query: Could not retrieve valid subgraph data.")
                return []
                
            subgraph_data = query_results[0]['subgraph_data']

            logger.debug(f"Dispatching GNN inference task to 'gnn_queue'...")
            
            task: AsyncResult = current_app.send_task(
                'phoenix.run_gnn_inference',
                args=[subgraph_data],
                queue='gnn_queue'
            )

            try:
                gnn_results = await asyncio.to_thread(task.get, timeout=30)
            except Exception as task_err:
                logger.error(f"GNN inference task failed or timed out: {task_err}")
                return []

            if not gnn_results or 'node_embeddings' not in gnn_results:
                logger.info("GNN inference returned no results (or model not loaded).")
                return []

            gnn_evidence_list = [
                Evidence(
                    id=f"gnn_node_{i}",
                    content=f"GNN processed node {i} from query '{query_text}'",
                    score=0.99,
                    source_type=DataSource.GRAPH,
                    metadata={"source": "knowledge_graph_gnn"}
                ) for i in range(self.top_k_gnn)
            ]
            return gnn_evidence_list.copy()

        except (ConnectionError, TimeoutError) as e:
            logger.error(
                f"[RAG Retrieval Failure] Connection/Timeout error in GNN-enhanced query for '{query_text}': {e}",
                exc_info=True
            )
            return []
        except Exception as e:
            logger.error(f"[RAG Retrieval Failure] Failed GNN-enhanced query for '{query_text}': {e}", exc_info=True)
            return []

    async def search_temporal_db(self, query: str) -> List[Evidence]:
        if not self.temporal_db:
            return []
        try:
            results = await self.temporal_db.search_events(query_string=query, size=5)

            evidence_list = []
            for event_data, score in results:
                evidence_list.append(Evidence(
                    id=event_data.get('id', 'temp_unknown'),
                    content=event_data.get('content') or event_data.get('summary') or str(event_data),
                    score=float(score),
                    source_type=DataSource.TEMPORAL,
                    metadata={"timestamp": event_data.get('timestamp')}
                ))
            
            logger.info(f"Temporal search found {len(evidence_list)} results.")
            return evidence_list
        except (ConnectionError, TimeoutError) as e:
            logger.error(
                f"[RAG Retrieval Failure] Connection/Timeout error during temporal search: {e}",
                exc_info=True
            )
            return []
        except Exception as e:
            logger.error(
                f"[RAG Retrieval Failure] Temporal search failed: {e}",
                exc_info=True
            )
            return []

    async def search_tabular_db(self, query: str) -> List[Evidence]:
        if not self.tabular_db:
            return []
        try:
            response = await self.tabular_db.query(query)
            results = response.get("results", [])

            evidence_list = []
            for i, row in enumerate(results):
                evidence_list.append(Evidence(
                    id=f"sql_res_{i}",
                    content=str(row),
                    score=1.0, 
                    source_type=DataSource.STRUCTURED,
                    metadata={"sql_query": response.get("generated_sql")}
                ))
            
            logger.info(f"Tabular search found {len(evidence_list)} results.")
            return evidence_list
        except (ConnectionError, TimeoutError) as e:
            logger.error(
                f"[RAG Retrieval Failure] Connection/Timeout error during tabular search: {e}",
                exc_info=True
            )
            return []
        except Exception as e:
            logger.error(
                f"[RAG Retrieval Failure] Tabular search failed: {e}",
                exc_info=True
            )
            return []

    async def search_web(self, query: str) -> List[Evidence]:
        # [Fix IV.2] 实现 Web 搜索
        if not self.search_client: return []
        logger.info(f"Performing web search for: {query}")
        try:
            # [Task IV.2] Updated to support both Async (Gemini) and Sync (Tavily) clients
            search_limit = self.config.get("l1_retriever", {}).get("search_top_k", 2)
            
            if asyncio.iscoroutinefunction(self.search_client.search):
                # Gemini Adapter is async
                response = await self.search_client.search(query=query, max_results=search_limit)
            else:
                # Old Tavily client is sync, requires thread wrapping
                response = await asyncio.to_thread(
                    self.search_client.search,
                    query=query,
                    max_results=search_limit
                )

            results = response.get("results", [])
            evidence_list = []
            for res in results:
                evidence_list.append(Evidence(
                    id=res.get("url", "web_unknown"),
                    content=res.get("content", ""),
                    score=0.7, # Web 结果的默认置信度
                    source_type=DataSource.WEB if hasattr(DataSource, 'WEB') else "web",
                    metadata={"title": res.get("title"), "url": res.get("url")}
                ))
            logger.info(f"Web search found {len(evidence_list)} results.")
            return evidence_list
        except (ConnectionError, TimeoutError) as e:
            logger.error(
                f"[RAG Retrieval Failure] Connection/Timeout error during web search: {e}",
                exc_info=True
            )
            return []
        except Exception as e:
            logger.error(
                f"[RAG Retrieval Failure] Web search failed: {e}",
                exc_info=True
            )
            return []

    async def retrieve_and_rerank(self, query: str) -> List[Evidence]:
        """
        Orchestrates the retrieval and reranking process.
        [Refactored for GNN Plan Task 2.3] Now runs sources in parallel.
        """
        logger.info(f"Starting retrieval for query: {query}")

        tasks = [
            self.search_vector_db(query),
            self.search_knowledge_graph(query), 
            self._query_graph_with_gnn(query, k=self.top_k_gnn), 
            self.search_temporal_db(query),
            self.search_tabular_db(query),
            self.search_web(query) 
        ]
        
        try:
            all_results = await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(
                f"[RAG Retrieval Failure] Critical error during asyncio.gather in retrieve_and_rerank: {e}",
                exc_info=True
            )
            all_results = [[], [], [], [], [], []] 

        vector_results: List[Evidence] = all_results[0]
        graph_search_results: List[GraphQueryResult] = all_results[1]
        gnn_results: List[Evidence] = all_results[2]
        temporal_results: List[Evidence] = all_results[3]
        tabular_results: List[Evidence] = all_results[4]
        web_results: List[Evidence] = all_results[5]

        successful_sources_count = sum(1 for result_list in all_results if result_list)
        
        if successful_sources_count < self.min_successful_sources:
            logger.critical(
                f"[RAG Completeness Failure] Failed to retrieve minimum context. "
                f"Sources Found: {successful_sources_count}, "
                f"Minimum Required: {self.min_successful_sources}. "
                f"Query: '{query}'. Aborting L1 Agent task.",
                exc_info=False
            )
            raise CognitiveError(f"Failed to retrieve minimum required context sources.")

        graph_results: List[Evidence] = [
            Evidence(
                id=res.node_id,
                content=res.content,
                score=res.score,
                source_type=res.source_type,
                metadata={"node_type": res.node_type},
            )
            for res in graph_search_results
        ]

        combined_results = (
            vector_results + 
            graph_results + 
            gnn_results +
            temporal_results +
            tabular_results +
            web_results
        )
        if not combined_results:
            logger.warning(f"No results found for query: {query}")
            return []

        logger.info(
            f"Combined {len(vector_results)} vector, "
            f"{len(graph_results)} graph, "
            f"{len(gnn_results)} GNN, "
            f"{len(temporal_results)} temporal, "
            f"{len(tabular_results)} tabular, and "
            f"{len(web_results)} web results."
        )

        final_results = self._rerank_documents(query, combined_results)

        final_top_k = self.config.get("l1_retriever", {}).get("final_top_k", 5)
        final_results = final_results[:final_top_k]

        logger.info(
            f"Retrieval complete. Returning {len(final_results)} reranked documents."
        )
        return final_results

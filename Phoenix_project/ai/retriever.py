"""
Advanced Retrieval module for the Phoenix project.

Handles hybrid search (vector, keyword, graph) and reranking.
"""

import logging
import asyncio
from typing import Any, List, Dict, Optional

from sentence_transformers import CrossEncoder  # type: ignore

from ..core.schemas.evidence_schema import (
    Evidence,
    QueryResult,
    GraphQueryResult,
    DataSource,
)
from ..memory.vector_store import VectorStore
from ..ai.graph_db_client import GraphDBClient
from ..api.gemini_pool_manager import GeminiClient
from ..ai.gnn_inferencer import GNNInferencer
# [Task 4] 导入 PromptManager 和 PromptRenderer
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
        llm_client: GeminiClient,
        config: Dict[str, Any],
        # [Task 4] 假设这些是由 Registry.py 注入的
        prompt_manager: PromptManager,
        prompt_renderer: PromptRenderer,
        gnn_inferencer: GNNInferencer,  # [GNN Plan Task 2.1]
        # --- 添加/更新这些参数 喵! ---
        temporal_db: Optional[TemporalDBClient] = None,
        tabular_db: Optional[TabularDBClient] = None,
        search_client: Optional[Any] = None,
        ensemble_client: Optional[EnsembleClient] = None
    ):
        """
        Initializes the Retriever.

        Args:
            vector_store: An instance of VectorStore.
            graph_db: An instance of GraphDBClient.
            llm_client: An instance of the GeminiClient for LLM operations.
            config: Configuration dictionary.
            prompt_manager: [Task 4] Injected PromptManager.
            prompt_renderer: [Task 4] Injected PromptRenderer.
            gnn_inferencer: [GNN Plan Task 2.1] Injected GNNInferencer.
        """
        self.vector_store = vector_store
        self.graph_db = graph_db
        self.gnn_inferencer = gnn_inferencer  # [GNT Plan Task 2.1]
        self.llm_client = llm_client
        # --- 存储这些新客户端 喵! ---
        self.temporal_db = temporal_db
        self.tabular_db = tabular_db
        self.search_client = search_client
        self.ensemble_client = ensemble_client # 存储

        # [Task 4] 存储注入的依赖
        self.prompt_manager = prompt_manager
        self.prompt_renderer = prompt_renderer
        
        # [修复] 存储 'ai_components.retriever' 根配置
        self.config = config 

        # [修复] 
        default_l1_config = self.config.get("l1_retriever", {})

        self.rerank_model_name = default_l1_config.get(
            "rerank_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self.top_k_vector = default_l1_config.get("vector_top_k", 10)
        self.top_k_graph = default_l1_config.get("top_k_graph", 5)
        self.top_k_gnn = default_l1_config.get("top_k_gnn", 5)
        self.rerank_threshold = default_l1_config.get("rerank_threshold", 0.1)

        self._initialize_reranker()

    def _initialize_reranker(self):
        """
        Initializes the CrossEncoder model for reranking.
        Includes a fallback mechanism.
        """
        try:
            self.reranker = CrossEncoder(self.rerank_model_name)
            logger.info(f"Successfully loaded reranker model: {self.rerank_model_name}")
        except Exception as e:
            logger.error(
                f"Failed to load CrossEncoder model '{self.rerank_model_name}'. "
                f"Error: {e}. Falling back to simple reranker."
            )
            self.reranker = None  # Flag to use fallback

    def _fallback_reranker(
        self, query: str, documents: List[Evidence]
    ) -> List[Evidence]:
        """
        A simple fallback reranker if the CrossEncoder model fails to load.
        """
        logger.warning("Using fallback reranker (no reranking).")
        # Just return top N documents without reranking
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
            # 使用回退重排器（如果 CrossEncoder 加载失败）
            return self._fallback_reranker(query, documents)

        try:
            # 准备 CrossEncoder 的输入
            pairs: List[List[str]] = [[query, doc.content] for doc in documents]
            if not pairs:
                return []

            # 计算分数
            scores = self.reranker.predict(pairs)

            # 更新文档分数并过滤
            reranked_docs: List[Evidence] = []
            for i, doc in enumerate(documents):
                new_score = float(scores[i])
                if new_score >= self.rerank_threshold:
                    doc.score = new_score  # 更新为 reranker 的分数
                    doc.source_type = DataSource.RERANKED_VECTOR
                    reranked_docs.append(doc)

            # 按新分数排序
            reranked_docs.sort(key=lambda x: x.score, reverse=True)

            logger.info(
                f"Reranked {len(documents)} docs down to {len(reranked_docs)}."
            )
            return reranked_docs

        except Exception as e:
            logger.error(f"Error during document reranking: {e}")
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
        except Exception as e:
            logger.error(f"Error during vector search: {e}")
            return []

    def _generate_cypher_prompt(self, query: str, schema: str) -> str:
        """
        [Task 4] 重构：使用 PromptRenderer 生成 Cypher 提示。
        """
        # 优化：[Task 4] 这个模板现在从 PromptManager 加载
        logger.debug("Rendering text_to_cypher prompt...")
        
        try:
            context = {
                "schema": schema,
                "query": query
            }
            
            # 1. 渲染整个模板结构
            # [Task 4] get_prompt 应该返回原始 dict
            rendered_data = self.prompt_renderer.render("text_to_cypher", context)
            
            # 2. 提取最终的提示字符串
            # (基于我们的 text_to_cypher.json 结构)
            prompt_str = rendered_data.get("full_prompt_template")
            if not prompt_str:
                 raise ValueError("'full_prompt_template' key not found in rendered prompt.")
            
            return prompt_str

        except Exception as e:
            logger.error(f"Failed to render text_to_cypher prompt: {e}. Using fallback.", exc_info=True)
            # [Task 4] 回退到旧的硬编码逻辑
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
            schema = await self.graph_db.get_schema() # 假设 graph_db 有这个方法
            if not schema:
                logger.warning("Could not retrieve graph schema for text-to-Cypher.")
                schema = "No schema available." # 继续尝试

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

        except Exception as e:
            logger.error(f"Error in _generate_text_to_cypher: {e}")
            return None


    async def search_knowledge_graph(self, query: str) -> List[GraphQueryResult]:
        """
        OPTIMIZED: Performs a search in the Knowledge Graph.
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

        except Exception as e:
            logger.error(f"Error during graph search with query '{cypher_query}': {e}")
            return []

    # [GNN Plan Task 2.2] New GNN-enhanced query method
    async def _query_graph_with_gnn(self, query_text: str, k: int) -> List[Evidence]:
        """
        Performs a GNN-enhanced query on the knowledge graph.

        Implements refinements:
        1. (Robustness) Top-level try/except to prevent asyncio.gather failure.
        2. (Performance) Assumes a single, efficient Cypher query.
        """
        # [Refinement Point 2] Top-level error handling for robustness
        try:
            logger.debug(f"Performing GNN-enhanced graph query for: {query_text}")

            # [Refinement Point 1]
            # Step A: Fetch Subgraph with a single query.
            # This query should find seed nodes AND fetch their N-hop neighbors
            # in one database operation.
            # (Placeholder query for now)
            cypher_query = """
            // Placeholder: Future query will find seeds and expand.
            MATCH (n)
            WHERE n.name IS NOT NULL
            RETURN {
                nodes: collect(DISTINCT n),
                edges: [] 
            } AS subgraph_data
            LIMIT 1
            """
            params = {"query": query_text, "k_seeds": k}
            
            # [Future Implementation] This logic will be more complex
            query_results = await self.graph_db.execute_query(cypher_query, params)
            
            if not query_results or 'subgraph_data' not in query_results[0]:
                logger.warning("GNN query: Could not retrieve valid subgraph data.")
                return []
                
            subgraph_data = query_results[0]['subgraph_data']

            # Step B: Call GNN Inference
            gnn_results = await self.gnn_inferencer.infer(graph_data=subgraph_data)

            # Step C: Process Results
            if not gnn_results or 'node_embeddings' not in gnn_results:
                logger.info("GNN inference returned no results or model not loaded.")
                return []

            # [Future Implementation] Parse actual gnn_results.
            # For now, create placeholder evidence.
            gnn_evidence_list = [
                Evidence(
                    id=f"gnn_node_{i}",
                    content=f"GNN processed node {i} from query '{query_text}'",
                    score=0.99,  # GNN results get high initial score
                    source_type=DataSource.GRAPH,
                    metadata={"source": "knowledge_graph_gnn"} # Differentiate
                ) for i in range(self.top_k_gnn) # Use self.top_k_gnn
            ]
            return gnn_evidence_list.copy()

        except Exception as e:
            logger.error(f"Failed GNN-enhanced query for '{query_text}': {e}", exc_info=True)
            return [] # Return empty list to not break asyncio.gather

    # --- 新的 RAG 搜索方法 (占位符) 喵! ---

    async def search_temporal_db(self, query: str) -> List[Evidence]:
        if not self.temporal_db: return []
        logger.info("TemporalDB search not yet implemented.")
        return [] # TODO: 实现时序检索

    async def search_tabular_db(self, query: str) -> List[Evidence]:
        if not self.tabular_db: return []
        logger.info("TabularDB search not yet implemented.")
        return [] # TODO: 实现表格检索 (e.g., Text-to-SQL)

    async def search_web(self, query: str) -> List[Evidence]:
        if not self.search_client: return []
        logger.info("Web search not yet implemented.")
        return [] # TODO: 实现网络搜索

    async def retrieve_and_rerank(self, query: str) -> List[Evidence]:
        """
        Orchestrates the retrieval and reranking process.
        [Refactored for GNN Plan Task 2.3] Now runs sources in parallel.
        """
        logger.info(f"Starting retrieval for query: {query}")

        # [GNN Plan Task 2.3] Run all retrieval tasks in parallel
        tasks = [
            self.search_vector_db(query),
            self.search_knowledge_graph(query), # Returns List[GraphQueryResult]
            self._query_graph_with_gnn(query, k=self.top_k_gnn), # Returns List[Evidence]
            # --- 添加新任务 喵! ---
            self.search_temporal_db(query),
            self.search_tabular_db(query),
            self.search_web(query)
        ]
        
        try:
            # Our _query_graph_with_gnn is already robust (returns [] on error)
            # search_vector_db and search_knowledge_graph also have internal
            # try/except blocks that return [], so this is safe.
            all_results = await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Critical error during parallel retrieval: {e}", exc_info=True)
            all_results = [[], [], [], [], [], []] # Empty results for each task

        # Unpack results
        vector_results: List[Evidence] = all_results[0]
        graph_search_results: List[GraphQueryResult] = all_results[1]
        gnn_results: List[Evidence] = all_results[2]
        # --- 解包新任务 喵! ---
        temporal_results: List[Evidence] = all_results[3]
        tabular_results: List[Evidence] = all_results[4]
        web_results: List[Evidence] = all_results[5]

        # Convert graph results to Evidence schema
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

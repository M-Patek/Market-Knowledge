from typing import List, Dict, Any, Optional
from Phoenix_project.ai.embedding_client import EmbeddingClient
from Phoenix_project.memory.vector_store import VectorStore
from Phoenix_project.memory.cot_database import CoTDatabase
# [主人喵的修复 1] 导入 KGS 以实现混合检索
from Phoenix_project.knowledge_graph_service import KnowledgeGraphService
from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

class Retriever:
    """
    [主人喵的修复 1]
    执行混合检索 (Hybrid Retrieval)。
    从 VectorStore (语义), CoTDatabase (历史), 和
    KnowledgeGraphService (结构化/GNN) 检索相关信息，
    以构建用于 AI 代理的上下文。
    """

    def __init__(
        self,
        vector_store: VectorStore,
        cot_database: CoTDatabase,
        embedding_client: EmbeddingClient,
        knowledge_graph_service: KnowledgeGraphService # <-- [主人喵的修复 1] 添加 KGS
    ):
        self.vector_store = vector_store
        self.cot_database = cot_database
        self.embedding_client = embedding_client
        self.knowledge_graph_service = knowledge_graph_service # <-- [主人喵的修复 1]
        logger.info("Retriever (Hybrid) initialized with KG support.")

    async def retrieve(
        self,
        query: str,
        target_symbols: List[str] = None,
        top_k_vector: int = 5,
        top_k_cot: int = 3
    ) -> Dict[str, List[Any]]:
        """
        [主人喵的修复 1]
        从所有可用存储中检索上下文。
        (重命名自 retrieve_relevant_context 以匹配 ReasoningEnsemble)
        """
        logger.info(f"Retrieving context for query: {query[:50]}... Symbols: {target_symbols}")
        
        metadata_filter = None
        if target_symbols:
            # (假设元数据过滤器使用 'symbol' 字段)
            metadata_filter = {"symbol": {"$in": target_symbols}} 

        # 1. Generate embedding for the query
        try:
            query_embedding = await self.embedding_client.get_embedding(query)
            if not query_embedding:
                logger.error("Failed to generate query embedding. Cannot perform vector search.")
                vector_chunks = []
            else:
                # 2. Query VectorStore (使用 metadata_filter)
                vector_chunks = await self.vector_store.search(
                    query_embedding=query_embedding,
                    top_k=top_k_vector,
                    metadata_filter=metadata_filter # <-- 使用过滤器
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

        # 4. Query KnowledgeGraphService
        try:
            # [主人喵的修复 1] 调用 KGS 的异步 query 方法
            kg_results = await self.knowledge_graph_service.query(query)
            logger.info(f"Retrieved {len(kg_results)} results from KnowledgeGraph.")
            
        except Exception as e:
            logger.error(f"Error during KnowledgeGraph retrieval: {e}", exc_info=True)
            kg_results = []

        return {
            "vector_chunks": vector_chunks,
            "cot_traces": cot_traces,
            "kg_results": kg_results # <-- [主人喵的修复 1] 添加图谱结果
        }

    def format_context(
        self, 
        retrieved_data: Dict[str, List[Any]], 
        max_tokens: int = 4096 # (合理的默认值)
    ) -> str:
        """
        [主人喵的修复 1]
        将检索到的数据 (KG, CoT, Vectors) 格式化为字符串
        以适应 LLM 提示的特定 Token 限制。
        (逻辑提取自旧的 retrieve_for_context_window)
        """
        
        formatted_context = "--- Relevant Knowledge ---\n\n"
        
        # 1. Add KG results (highest priority)
        for item in retrieved_data.get("kg_results", []):
            # (将 dict 转换为更易读的字符串)
            kg_text = f"Knowledge Graph Fact (Source: {item.get('source', 'KG')}):\n{str(item)}\n\n"
            if len(formatted_context) + len(kg_text) > max_tokens:
                break
            formatted_context += kg_text
        
        # 2. Add CoT traces (high-value)
        for trace in retrieved_data.get("cot_traces", []):
            trace_text = f"Previous Reasoning ({trace.get('timestamp', 'N/A')}):\n{trace.get('reasoning', '')}\nDecision: {trace.get('decision', '')}\n\n"
            if len(formatted_context) + len(trace_text) > max_tokens:
                break
            formatted_context += trace_text
            
        # 3. Add vector chunks
        for chunk in retrieved_data.get("vector_chunks", []):
            chunk_text = f"Retrieved Document (Source: {chunk.get('source', 'N/A')}):\n{chunk.get('text', '')}\n\n"
            if len(formatted_context) + len(chunk_text) > max_tokens:
                break
            formatted_context += chunk_text

        if len(formatted_context) > max_tokens:
             formatted_context = formatted_context[:max_tokens] + "... [Truncated]"
             
        logger.info(f"Assembled context window of {len(formatted_context)} chars.")
        return formatted_context

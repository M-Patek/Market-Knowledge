import asyncio
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import CrossEncoder
import json

from Phoenix_project.ai.embedding_client import EmbeddingClient
from Phoenix_project.memory.vector_store import BaseVectorStore, Document
from Phoenix_project.memory.cot_database import CoTDatabase
from Phoenix_project.knowledge_graph_service import KnowledgeGraphService
from Phoenix_project.monitor.logging import get_logger
from Phoenix_project.config.loader import ConfigLoader
from Phoenix_project.ai.temporal_db_client import TemporalDBClient
from Phoenix_project.ai.tabular_db_client import TabularDBClient

logger = get_logger(__name__)

class Retriever:
    """
    [蓝图 2 已更新]
    执行先进的混合检索流程 (Hybrid Retrieval)。
    1. 并行从 VectorStore, TemporalDB, 和 TabularDB 检索。
    2. 使用 RRF (Reciprocal Rank Fusion) 融合排名。
    3. 使用 Cross-Encoder (重排模型) 对融合后的列表进行重排。
    4. 格式化 Top-N 结果以供 LLM 使用。
    """

    def __init__(
        self,
        config_loader: ConfigLoader, # <--- [蓝图 2] 添加
        vector_store: BaseVectorStore, # <--- [蓝图 2] 更改为 BaseVectorStore
        cot_database: CoTDatabase,
        embedding_client: EmbeddingClient,
        knowledge_graph_service: KnowledgeGraphService,
        temporal_client: TemporalDBClient, # <--- [蓝图 2] 添加
        tabular_client: TabularDBClient  # <--- [蓝图 2] 添加
    ):
        self.vector_store = vector_store
        self.cot_database = cot_database
        self.embedding_client = embedding_client
        self.knowledge_graph_service = knowledge_graph_service
        self.temporal_client = temporal_client
        self.tabular_client = tabular_client
        
        # [蓝图 2] 加载配置和重排模型
        try:
            system_config = config_loader.load_config('config/system.yaml')
            retriever_config = system_config.get("ai", {}).get("retriever", {})
            rerank_config = retriever_config.get("rerank", {})
            
            rerank_model_name = rerank_config.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            self.reranker = CrossEncoder(rerank_model_name)
            
            self.rerank_top_n = rerank_config.get("top_n", 10)
            self.rrf_k = retriever_config.get("rrf_k", 60)
            
            logger.info(f"Retriever (Advanced RAG) initialized. Reranker model: {rerank_model_name}")
        except Exception as e:
            logger.error(f"Failed to load CrossEncoder model, using fallback. Error: {e}", exc_info=True)
            # 提供一个回退的 reranker，以防模型加载失败
            self.reranker = None
            self.rerank_top_n = 10
            self.rrf_k = 60


    async def retrieve(
        self,
        query: str,
        target_symbols: List[str] = None,
        top_k_vector: int = 10,
        top_k_temporal: int = 10,
        top_k_tabular: int = 10
    ) -> Dict[str, List[Any]]:
        """
        [蓝图 2 已重构]
        从所有来源并行检索带分数的文档列表。
        (替换了旧的 retrieve 方法)
        """
        logger.info(f"Retrieving context for query: {query[:50]}... Symbols: {target_symbols}")
        
        metadata_filter = None
        if target_symbols:
            metadata_filter = {"symbol": {"$in": target_symbols}} 

        # 1. 并行检索任务
        tasks = []
        
        # a. 向量存储 (语义搜索)
        tasks.append(self.vector_store.asimilarity_search(
            query=query,
            k=top_k_vector,
            filter=metadata_filter # 假设 pinecone/mock store 支持 'filter'
        ))
        
        # b. 时序存储 (关键字/时间搜索)
        tasks.append(self.temporal_client.search_events(
            symbols=target_symbols,
            query_string=query,
            size=top_k_temporal
        ))
        
        # c. 表格存储 (结构化搜索)
        tasks.append(self.tabular_client.search_financials(
            query=query,
            symbols=target_symbols,
            limit=top_k_tabular
        ))

        # 2. 执行所有检索
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            vector_results_scored = results[0] if not isinstance(results[0], Exception) else []
            temporal_results_scored = results[1] if not isinstance(results[1], Exception) else []
            tabular_results_scored = results[2] if not isinstance(results[2], Exception) else []

            if isinstance(results[0], Exception):
                logger.error(f"Vector search failed: {results[0]}", exc_info=results[0])
            if isinstance(results[1], Exception):
                logger.error(f"Temporal search failed: {results[1]}", exc_info=results[1])
            if isinstance(results[2], Exception):
                logger.error(f"Tabular search failed: {results[2]}", exc_info=results[2])

            logger.info(f"Retrieved: {len(vector_results_scored)} vector, "
                        f"{len(temporal_results_scored)} temporal, "
                        f"{len(tabular_results_scored)} tabular docs.")

        except Exception as e:
            logger.error(f"Error during parallel retrieval: {e}", exc_info=True)
            vector_results_scored, temporal_results_scored, tabular_results_scored = [], [], []

        # 3. [蓝图 2] 返回带分数的文档列表以供 RRF 使用
        # (我们保留旧的 CoT 和 KG 查询，以防 format_context 仍在使用它们，
        # 尽管新的 RAG 流程不使用它们)
        try:
            cot_traces = await self.cot_database.search_traces(keywords=query.split(), limit=3)
        except Exception: cot_traces = []
        
        try:
            kg_results = await self.knowledge_graph_service.query(query)
        except Exception: kg_results = []
        

        return {
            "vector_chunks_scored": vector_results_scored,
            "temporal_chunks_scored": temporal_results_scored,
            "tabular_chunks_scored": tabular_results_scored,
            "cot_traces": cot_traces, # 保留以实现向后兼容
            "kg_results": kg_results  # 保留以实现向后兼容
        }

    def _reciprocal_rank_fusion(
        self, 
        ranked_lists: List[List[Tuple[Any, float]]], 
        k: int = 60
    ) -> List[Tuple[Any, float]]:
        """
        [蓝图 2]
        执行 RRF (Reciprocal Rank Fusion)。
        
        Args:
            ranked_lists: [ [ (doc_A, score1), (doc_B, score2) ], [ (doc_B, score3), (doc_C, score4) ] ]
                          文档对象 (Any) 可以是 Document、dict 等。
            k (int): RRF 使用的 k 常量。

        Returns:
            List[Tuple[Any, float]]: 按 RRF 分数排序的 (doc, rrf_score) 元组列表。
        """
        if not ranked_lists:
            return []
            
        # 1. 规范化分数并计算 RRF
        # 我们需要一个唯一的方式来标识文档 (例如，id 或内容的哈希值)
        def get_doc_id(doc):
            if isinstance(doc, Document):
                # 尝试 'id'，如果不存在则回退到哈希内容
                doc_id = doc.metadata.get("id")
                if doc_id:
                    return doc_id
                return hash(doc.page_content)
            if isinstance(doc, dict):
                # 尝试 'id'，如果不存在则回退到哈希 JSON
                doc_id = doc.get("id")
                if doc_id:
                    return doc_id
                try:
                    return hash(json.dumps(doc, sort_keys=True))
                except TypeError:
                    # 如果字典包含不可哈希的类型，则回退到 str
                    return hash(str(doc))
            return hash(doc) # 回退

        rrf_scores = {}
        doc_store = {} # 存储 doc_id -> doc 对象的映射

        for doc_list in ranked_lists:
            # 确保我们有分数
            if not doc_list or not isinstance(doc_list[0], (list, tuple)) or len(doc_list[0]) < 2:
                logger.warning(f"RRF skipping invalid or empty list: {str(doc_list)[:100]}...")
                continue

            # (可选) 重新排序（如果列表未排序）
            # sorted_list = sorted(doc_list, key=lambda x: x[1], reverse=True)
            sorted_list = doc_list # 假设已排序

            for rank, (doc, score) in enumerate(sorted_list):
                if score is None or score <= 0: # RRF 忽略 0、None 或负分
                    continue
                    
                doc_id = get_doc_id(doc)
                if doc_id not in doc_store:
                    doc_store[doc_id] = doc
                
                # RRF 公式
                rank_score = 1.0 / (k + rank + 1)
                
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0.0
                rrf_scores[doc_id] += rank_score

        # 2. 转换为排序列表
        fused_results = []
        for doc_id, score in rrf_scores.items():
            fused_results.append((doc_store[doc_id], score))
            
        fused_results.sort(key=lambda x: x[1], reverse=True)
        
        return fused_results

    def _doc_to_string(self, doc: Any) -> str:
        """辅助函数：将任何检索到的文档转换为用于重排的字符串。"""
        if isinstance(doc, Document):
            return doc.page_content
        if isinstance(doc, dict):
            # 适用于时序数据 (dict) 或表格数据 (dict)
            try:
                return json.dumps(doc)
            except TypeError:
                return str(doc) # 回退
        return str(doc)

    def format_context(
        self, 
        query: str, # <--- [蓝图 2] 需要 query
        retrieved_data: Dict[str, List[Any]], 
        max_tokens: int = 8192 # (合理的默认值)
    ) -> str:
        """
        [蓝图 2 已重构]
        替换“简单上下文组装”。
        执行 RRF、Cross-Encoder 重排，并格式化最终的 Top-N 上下文。
        """
        
        # 1. 从检索到的数据中提取带分数的列表
        vector_list = retrieved_data.get("vector_chunks_scored", [])
        temporal_list = retrieved_data.get("temporal_chunks_scored", [])
        tabular_list = retrieved_data.get("tabular_chunks_scored", [])

        # 2. [蓝图 2] 融合 (RRF)
        fused_results = self._reciprocal_rank_fusion(
            [vector_list, temporal_list, tabular_list],
            k=self.rrf_k
        )
        
        if not fused_results:
            logger.warning("RRF returned no results. Context will be empty.")
            return "--- Relevant Knowledge ---\n\nNo relevant documents found.\n"

        # 3. [蓝图 2] 重排 (Cross-Encoder)
        if self.reranker:
            try:
                # 准备用于重排的 (query, doc_string) 对
                # 我们只重排 RRF 后的 Top-K (例如 Top 50)
                rerank_candidates = [doc for doc, score in fused_results[:50]]
                
                sentence_pairs = [
                    (query, self._doc_to_string(doc)) for doc in rerank_candidates
                ]
                
                if sentence_pairs:
                    logger.info(f"Reranking {len(sentence_pairs)} candidates with CrossEncoder...")
                    # 计算分数
                    cross_scores = self.reranker.predict(sentence_pairs)
                    
                    # 组合
                    reranked_results = list(zip(rerank_candidates, cross_scores))
                    # 按 cross_scores 降序排序
                    reranked_results.sort(key=lambda x: x[1], reverse=True)
                else:
                    reranked_results = []
                    
            except Exception as e:
                logger.error(f"Cross-encoder reranking failed: {e}. Falling back to RRF order.", exc_info=True)
                # 回退：使用 RRF 结果
                reranked_results = fused_results 
        else:
            logger.warning("No Reranker loaded. Using RRF order.")
            reranked_results = fused_results

        # 4. [蓝图 2] 格式化 Top-N
        final_docs = [doc for doc, score in reranked_results[:self.rerank_top_n]]
        
        formatted_context = "--- Relevant Knowledge (Ranked) ---\n\n"
        current_len = 0
        
        for i, doc in enumerate(final_docs):
            doc_str = self._doc_to_string(doc)
            
            # (从 Document 或 dict 中提取来源)
            source = "Unknown"
            if isinstance(doc, Document):
                source = doc.metadata.get("source", "VectorDB")
            elif isinstance(doc, dict):
                # 尝试从时序或表格数据中获取来源
                source = doc.get("source", doc.get("metric_name", "TabularDB"))
                if source == "TabularDB" and "metric_name" in doc:
                     source = f"TabularDB (Metric: {doc.get('metric_name')}, Symbol: {doc.get('symbol')})"
                elif "headline" in doc:
                     source = f"TemporalDB (Event: {doc.get('headline', 'N/A')[:30]}...)"


            doc_text = f"--- Document {i+1} (Source: {source}) ---\n"
            doc_text += doc_str + "\n\n"
            
            # (使用简单的字符长度检查来估算 token)
            if current_len + len(doc_text) > max_tokens:
                formatted_context += "... [Truncated due to token limit]\n"
                break
                
            formatted_context += doc_text
            current_len += len(doc_text)

        logger.info(f"Assembled advanced context window of {current_len} chars ({len(final_docs)} docs).")
        return formatted_context

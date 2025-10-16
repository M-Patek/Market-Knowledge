import logging
import asyncio
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime, date, timezone
import google.generativeai as genai
from pydantic import BaseModel, Field

from .vector_db_client import VectorDBClient
from .temporal_db_client import TemporalDBClient
from .tabular_db_client import TabularDBClient
from .relation_extractor import RelationExtractor, KnowledgeGraph
from .embedding_client import EmbeddingClient
from sentence_transformers import CrossEncoder

# --- 用于结构化LLM输出的Pydantic模型 ---
class DeconstructedQuery(BaseModel):
    """解构后用户查询的结构化表示。"""
    keywords: List[str] = Field(..., description="用于向量搜索的语义关键词列表。")
    ticker: Optional[str] = Field(None, description="提及的股票代码，如果有的话。")
    start_date: Optional[date] = Field(None, description="提及的日期范围的开始，如果有的话。")
    end_date: Optional[date] = Field(None, description="提及的日期范围的结束，如果有的话。")
    metric_name: Optional[str] = Field(None, description="提及的具体财务指标 (例如, '收入', '每股收益')。")

    class Config:
        # 为模型参考添加一个示例
        schema_extra = {"example": {"keywords": ["市场对Q3财报的反应", "苹果"], "ticker": "AAPL", "start_date": "2023-09-01", "end_date": "2023-09-30", "metric_name": "财报"}}


class HybridRetriever:
    """
    协调跨不同数据源（向量、时序、表格）的多路检索策略，
    并融合结果。
    """
    def __init__(self, vector_db_client: VectorDBClient, temporal_db_client: TemporalDBClient, tabular_db_client: TabularDBClient, rerank_config: Dict[str, Any]):
        self.logger = logging.getLogger("PhoenixProject.HybridRetriever")
        self.vector_client = vector_db_client
        self.temporal_client = temporal_db_client
        self.tabular_client = tabular_db_client
        self.embedding_client = EmbeddingClient()
        self.retriever_config = rerank_config
        self.rrf_k = self.retriever_config.get('rrf_k', 60)
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # [NEW] Initialize a separate, simple generative model for HyDE
        try:
            self.hyde_model = genai.GenerativeModel("gemini-1.5-flash-latest")
        except Exception as e:
            self.logger.error(f"Failed to initialize HyDE model: {e}")
            self.hyde_model = None
        
        # 初始化关系提取器
        self.relation_extractor = RelationExtractor()

        # 初始化用于查询解构的生成式模型
        try:
            # 将Pydantic模型作为工具，升级模型为"函数调用"模型
            self.deconstructor_model = genai.GenerativeModel(
                "gemini-1.5-flash-latest", tools=[DeconstructedQuery]
            )
        except Exception as e:
            self.logger.error(f"初始化解构器模型失败: {e}")
            self.deconstructor_model = None
        self.logger.info(f"HybridRetriever已初始化，RRF k={self.rrf_k}。")

    async def _recall_from_vector(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """从向量数据库召回。"""
        hyde_vector = None
        if self.hyde_model:
            try:
                # 1. Generate a hypothetical document
                prompt = f"Please write a short, hypothetical paragraph that would be the perfect answer to the financial query: '{query}'"
                response = await self.hyde_model.generate_content_async(prompt)
                hypothetical_doc = response.text
                self.logger.info(f"Generated Hypothetical Document for HyDE: '{hypothetical_doc[:100]}...'")
                # 2. Embed the hypothetical document
                hyde_vector = self.embedding_client.create_query_embedding(hypothetical_doc)
            except Exception as e:
                self.logger.error(f"HyDE generation failed: {e}. Falling back to standard query embedding.")

        # 3. Query using the HyDE vector, or fall back to the original query's vector
        return self.vector_client.query(query, top_k=top_k, query_vector=hyde_vector)

    async def _recall_from_temporal(self, ticker: str, start_date: date, end_date: date) -> List[Dict[str, Any]]:
        """从时序数据库召回。"""
        return self.temporal_client.query_by_date(ticker, start_date, end_date)

    async def _recall_from_tabular(self, ticker: str) -> List[Dict[str, Any]]:
        """从表格数据库召回。"""
        return self.tabular_client.query_by_ticker(ticker)

    async def _deconstruct_query(self, query: str) -> DeconstructedQuery:
        """
        使用专用于工具调用的生成式模型将自然语言查询分解
        为用于定向检索的结构化组件。
        """
        if not self.deconstructor_model:
            self.logger.warning("解构器模型不可用。使用简单关键词提取。")
            return DeconstructedQuery(keywords=query.split())
        
        prompt = f"请解构以下金融查询: '{query}'"
        
        try:
            response = await self.deconstructor_model.generate_content_async(
                prompt,
                tool_config={'function_calling_config': 'ANY'}
            )
            fc = response.candidates[0].content.parts[0].function_call
            deconstructed_args = {key: value for key, value in fc.args.items()}
            return DeconstructedQuery(**deconstructed_args)
        except Exception as e:
            self.logger.error(f"使用LLM解构查询失败: {e}。回退到简单提取。")
            return DeconstructedQuery(keywords=query.split())

    async def recall(self, query: str, ticker: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        并行地从所有可用数据源执行第一阶段检索，
        并使用倒数排序融合（RRF）进行融合。
        """
        self.logger.info(f"为查询启动召回阶段: '{query}'")
        keywords = query
        query_ticker = ticker
        
        tasks = []
        if self.vector_client:
            tasks.append(self._recall_from_vector(keywords, top_k=50)) # Increased for better reranking
        if self.temporal_client and query_ticker:
            end_date = date.today()
            start_date = date(end_date.year - 1, end_date.month, end_date.day)
            tasks.append(self._recall_from_temporal(query_ticker, start_date, end_date))
        if self.tabular_client and query_ticker:
            tasks.append(self._recall_from_tabular(query_ticker))

        results_from_all_indexes = await asyncio.gather(*tasks, return_exceptions=True)
        
        rrf_scores = {}
        master_doc_lookup = {}
        for result_set in results_from_all_indexes:
            if isinstance(result_set, list):
                for rank, doc in enumerate(result_set):
                    source_id = doc.get('source_id')
                    if not source_id: continue
                    if source_id not in master_doc_lookup or 'vector_similarity_score' in doc:
                        master_doc_lookup[source_id] = doc
                    rrf_scores[source_id] = rrf_scores.get(source_id, 0.0) + (1.0 / (self.rrf_k + rank + 1))

        fused_candidates = {}
        for source_id, score in rrf_scores.items():
            if source_id in master_doc_lookup:
                doc = master_doc_lookup[source_id]
                doc['rrf_score'] = score
                
                # [NEW] Generate and inject the data fingerprint
                content = doc.get('content', '')
                if content:
                    doc['data_fingerprint'] = hashlib.sha256(content.encode('utf-8')).hexdigest()
                    
                fused_candidates[source_id] = doc

        final_candidates = sorted(fused_candidates.values(), key=lambda x: x['rrf_score'], reverse=True)
        self.logger.info(f"召回阶段完成。使用RRF融合并排序了 {len(final_candidates)} 个唯一候选。")
        return final_candidates

    def rerank(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """使用Cross-Encoder模型对候选列表进行深度语义重排。"""
        if not candidates or not query:
            return candidates

        # 准备模型输入：[query, document_content]
        model_inputs = [[query, doc.get('content', '')] for doc in candidates]
        
        # 使用Cross-Encoder计算语义相关性分数
        self.logger.info(f"Reranking {len(model_inputs)} candidates with CrossEncoder...")
        semantic_scores = self.cross_encoder.predict(model_inputs, show_progress_bar=False)

        # 将分数添加到候选文档中
        for doc, score in zip(candidates, semantic_scores):
            doc['semantic_rerank_score'] = float(score)

        # 根据新的语义分数降序排序
        reranked_results = sorted(candidates, key=lambda x: x['semantic_rerank_score'], reverse=True)
        self.logger.info(f"使用Cross-Encoder重排了 {len(reranked_results)} 个候选。")
        return reranked_results

    async def retrieve(self, query: str, ticker: Optional[str] = None, top_k: int = 10) -> Dict[str, Any]:
        """
        执行完整的检索和重排流程的主公共方法。
        """
        self.logger.info(f"--- [混合检索器]: 为查询 '{query}' 启动完整检索流程 ---")
        recalled_candidates = await self.recall(query, ticker)
        reranked_results = self.rerank(query, recalled_candidates)
        final_top_k = reranked_results[:top_k]
        
        # 从最终证据中提取关系图
        evidence_texts = [doc.get('content', '') for doc in final_top_k if doc.get('content')]
        knowledge_graph = await self.relation_extractor.extract_graph(evidence_texts)

        self.logger.info(f"检索流程完成。返回前 {len(final_top_k)} 个结果和知识图谱。")
        
        return {
            "evidence_documents": final_top_k,
            "knowledge_graph": knowledge_graph.dict()
        }

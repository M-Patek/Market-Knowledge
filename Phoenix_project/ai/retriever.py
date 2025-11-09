import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import CrossEncoder # type: ignore
# 修复：将 'core.schemas...' 转换为 'Phoenix_project.core.schemas...'
from Phoenix_project.core.schemas.evidence_schema import (
    Evidence,
    FinancialsEvidence,
    TemporalEvidence,
    GraphEvidence,
    TextEvidence,
)
# 修复：将 'memory.vector_store' 转换为 'Phoenix_project.memory.vector_store'
from Phoenix_project.memory.vector_store import VectorStore
# 修复：将 'monitor.logging' 转换为 'Phoenix_project.monitor.logging'
from Phoenix_project.monitor.logging import get_logger
# 修复：导入此模块所需的依赖
from Phoenix_project.ai.embedding_client import EmbeddingClient
from Phoenix_project.ai.graph_db_client import GraphDBClient
from Phoenix_project.ai.tabular_db_client import TabularDBClient
from Phoenix_project.ai.temporal_db_client import TemporalDBClient
from Phoenix_project.config.loader import ConfigLoader # 假设 ConfigLoader 也被传入

log = get_logger("Retriever")


class Retriever:
    """
    数据检索器，负责从所有数据源（向量、时序、表格、图）中检索信息。
    """

    # 修复：更新 __init__ 签名以匹配 worker.py
    def __init__(
        self,
        config_loader: ConfigLoader, # 修复：添加 config_loader
        vector_store: VectorStore,
        cot_database: Any, # 修复：添加 cot_database (来自 worker.py)
        embedding_client: EmbeddingClient,
        knowledge_graph_service: Any, # 修复：更改为 kg_service
        temporal_client: TemporalDBClient, # 修复：重命名
        tabular_client: TabularDBClient # 修复：重命名
        # config: Dict[str, Any], # 修复：Config 现在通过 config_loader 获取
    ):
        self.vector_store = vector_store
        self.temporal_db = temporal_client # 修复：使用新名称
        self.tabular_db = tabular_client # 修复：使用新名称
        self.graph_db = knowledge_graph_service # 修复：使用新名称
        self.embedding_client = embedding_client
        self.cot_database = cot_database # 修复：存储
        self.config_loader = config_loader # 修复：存储
        
        # 修复：从 config_loader 加载 'ai' 部分
        self.config = self.config_loader.load_config('system.yaml').get('ai', {})
        
        self.reranker_model_name = self.config.get("retriever", {}).get( # 修复：更深层的配置嵌套
            "reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        try:
            self.reranker = CrossEncoder(self.reranker_model_name)
            log.info(f"Reranker model '{self.reranker_model_name}' loaded successfully.")
        except Exception as e:
            log.warning(
                f"Failed to load Reranker model '{self.reranker_model_name}'. "
                f"Reason: {e}. Falling back to keyword matching or simple ranking."
            )
            self.reranker = self._fallback_reranker

    def _fallback_reranker(self, query: str, texts: List[str]) -> List[float]:
        """
        一个回退的 reranker，以防 CrossEncoder 模型加载失败。
        """
        log.debug(
            f"Using fallback reranker for query: '{query}'"
        )
        scores = []
        query_words = set(query.lower().split())
        for text in texts:
            text_words = set(text.lower().split())
            common_words = query_words.intersection(text_words)
            score = len(common_words) / len(query_words) if query_words else 0
            scores.append(score)
        return scores

    # 修复：签名与 ai/retriever.py (新版) 不匹配
    # def search_financials(self, query: str, symbol: str) -> FinancialsEvidence:
    async def search_financials(self, query: str, symbol: str) -> FinancialsEvidence: # 修复：设为 async
        """
        检索表格财务数据。
        [✅ 优化] 此处逻辑已移至 tabular_db_client，由其决定使用 SQL 代理还是 ILIKE。
        """
        try:
            # 修复： tabular_db.search_financials 现在是 async
            results = await self.tabular_db.search_financials(query, symbol)
            return FinancialsEvidence(
                data=results,
                query=query,
                symbol=symbol,
                source="TabularDB",
            )
        except Exception as e:
            log.error(f"Error searching financials for {symbol} with query '{query}': {e}")
            return FinancialsEvidence(data=[], query=query, symbol=symbol, source="TabularDB", error=str(e))

    def _parse_date_range(self, query: str) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        [✅ 优化] 从查询中解析日期范围。
        这是一个简化的实现，可以根据需要扩展以支持更复杂的自然语言。
        """
        now = datetime.now()
        
        # 匹配 "last X days/weeks/months/years"
        last_match = re.search(r"last\s+(\d+)\s+(day|week|month|year)s?", query, re.IGNORECASE)
        if last_match:
            value = int(last_match.group(1))
            unit = last_match.group(2).lower()
            end_date = now
            if unit == "day":
                start_date = now - timedelta(days=value)
            elif unit == "week":
                start_date = now - timedelta(weeks=value)
            elif unit == "month":
                start_date = now - timedelta(days=value * 30)  # 简化
            elif unit == "year":
                start_date = now - timedelta(days=value * 365) # 简化
            else:
                start_date = None
            
            log.debug(f"Parsed date range: {start_date} to {end_date} from '{query}'")
            return start_date, end_date

        # 匹配 "from YYYY-MM-DD to YYYY-MM-DD"
        range_match = re.search(r"from\s+(\d{4}-\d{2}-\d{2})\s+to\s+(\d{4}-\d{2}-\d{2})", query, re.IGNORECASE)
        if range_match:
            try:
                start_date = datetime.fromisoformat(range_match.group(1))
                end_date = datetime.fromisoformat(range_match.group(2))
                log.debug(f"Parsed date range: {start_date} to {end_date} from '{query}'")
                return start_date, end_date
            except ValueError:
                pass

        log.debug(f"Could not parse specific date range from '{query}'. Using default (None, None).")
        # 默认返回 None，让 temporal_db_client 使用其默认值
        return None, None

    def _parse_metrics(self, query: str) -> Optional[List[str]]:
        """
        [✅ 优化] 从查询中解析指标。
        """
        metrics = []
        if "price" in query.lower() or "close" in query.lower():
            metrics.append("close")
        if "volume" in query.lower():
            metrics.append("volume")
        if "high" in query.lower():
            metrics.append("high")
        if "low" in query.lower():
            metrics.append("low")
        if "open" in query.lower():
            metrics.append("open")
        
        if metrics:
            log.debug(f"Parsed metrics: {metrics} from '{query}'")
            return list(set(metrics)) # 返回唯一值
        
        log.debug(f"Could not parse specific metrics from '{query}'. Using default (None).")
        return None # 默认返回 None，让 temporal_db_client 使用其默认值

    # 修复：签名与 ai/retriever.py (新版) 不匹配
    # def search_temporal_db(self, query: str, symbol: str) -> TemporalEvidence:
    async def search_temporal_db(self, query: str, symbol: str) -> TemporalEvidence: # 修复：设为 async
        """
        检索时序数据 (例如股价)。
        [✅ 优化] query 现在用于确定时间范围和指标。
        """
        try:
            # [✅ 优化] 解析查询
            start_date, end_date = self._parse_date_range(query)
            metrics = self._parse_metrics(query)

            # [✅ 优化] 使用解析出的参数调用时序数据库
            # 注意：如果解析为 (None, None)，temporal_db_client 将使用其默认范围
            # 修复： temporal_db.search_range 现在是 async
            df = await self.temporal_db.search_range(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                metrics=metrics,
            )
            
            return TemporalEvidence(
                data=df.to_dict("records"),
                query=query,
                symbol=symbol,
                source="TemporalDB",
                start_date=start_date,
                end_date=end_date,
            )
        except Exception as e:
            log.error(f"Error searching temporal for {symbol} with query '{query}': {e}")
            return TemporalEvidence(data=[], query=query, symbol=symbol, source="TemporalDB", error=str(e))

    # 修复：签名与 ai/retriever.py (新版) 不匹配
    # def search_vector_db(self, query: str, k: int) -> List[TextEvidence]:
    async def search_vector_db(self, query: str, k: int) -> List[TextEvidence]: # 修复：设为 async
        """
        从向量存储中检索非结构化文本。
        """
        try:
            # 修复： embedding_client.embed 现在是 async
            query_embedding_list = await self.embedding_client.embed([query])
            if not query_embedding_list:
                raise ValueError("Failed to embed query")
            query_embedding = query_embedding_list[0]
            
            # 修复： vector_store.search 现在是 async
            search_results = await self.vector_store.search(query_embedding, k=k)

            # 重新排序 (Reranking)
            if self.reranker and search_results: # 修复：检查 search_results
                
                # 修复：search_results 是 (Document, score) 的元组列表
                passages = [
                    result[0].page_content for result in search_results
                ]
                
                # [query, passage] pairs
                rerank_input = [
                    (query, passage) for passage in passages
                ]
                
                # 修复：CrossEncoder.predict 是同步的，在线程中运行
                scores = await asyncio.to_thread(self.reranker.predict, rerank_input)

                # 组合结果和分数
                reranked_results = [
                    (search_results[i][0], scores[i]) for i in range(len(search_results)) # (Document, new_score)
                ]
                # 按新分数排序
                reranked_results.sort(key=lambda x: x[1], reverse=True)
                
                # 提取排序后的结果
                # search_results = [res[0] for res in reranked_results] # 修复：这只提取了 Document
                search_results_docs = [res[0] for res in reranked_results]
                search_results_scores = [res[1] for res in reranked_results]
                log.debug(f"Reranked {len(search_results_docs)} results for query: '{query}'")

                # 修复：重建 TextEvidence
                return [
                    TextEvidence(
                        text=doc.page_content, # 修复：使用 page_content
                        source=doc.metadata.get("source", "VectorStore"), # 修复：从 metadata 获取
                        timestamp=doc.metadata.get("timestamp"), # 修复：从 metadata 获取
                        metadata=doc.metadata,
                        query=query,
                        score=float(score), # 修复：使用 reranked score
                    )
                    for doc, score in zip(search_results_docs, search_results_scores)
                ]

            # 修复：如果 reranker 未运行，处理 (Document, score) 元组
            return [
                TextEvidence(
                    text=result[0].page_content,
                    source=result[0].metadata.get("source", "VectorStore"),
                    timestamp=result[0].metadata.get("timestamp"),
                    metadata=result[0].metadata,
                    query=query,
                    score=float(result[1]), # 原始分数
                )
                for result in search_results
            ]
        except Exception as e:
            log.error(f"Error searching vector DB with query '{query}': {e}")
            return [TextEvidence(text="", source="VectorStore", query=query, error=str(e))]

    # 修复：签名与 ai/retriever.py (新版) 不匹配
    # def search_knowledge_graph(self, query: str) -> GraphEvidence:
    async def search_knowledge_graph(self, query: str) -> GraphEvidence: # 修复：设为 async
        """
        检索知识图谱。
        """
        try:
            # TODO: 实现一个 "text-to-Cypher" 的转换器
            # 这是一个简化的 CONTAINS 搜索，用于占位
            cypher_query = f"MATCH (n) WHERE n.name CONTAINS $query_param RETURN n LIMIT 10"
            params = {"query_param": query} # 修复：使用参数
            
            # 修复： graph_db (KnowledgeGraphService) .query 是 async
            results = await self.graph_db.query(cypher_query, params=params)
            
            return GraphEvidence(
                data=results, query=query, cypher_query=cypher_query, source="GraphDB"
            )
        except Exception as e:
            log.error(f"Error searching graph DB with query '{query}': {e}")
            return GraphEvidence(data=[], query=query, source="GraphDB", error=str(e))

    # 修复：签名与 ai/retriever.py (新版) 不匹配
    # def retrieve_evidence(
    async def retrieve_evidence( # 修复：设为 async
        self, query: str, symbol: str, data_types: List[str], k_vector: int
    ) -> List[Evidence]:
        """
        根据数据类型检索所有相关证据。
        """
        evidence_list = []
        tasks = []
        
        # 修复：并行执行所有检索
        
        if "financials" in data_types:
            tasks.append(self.search_financials(query, symbol))
        if "temporal" in data_types:
            tasks.append(self.search_temporal_db(query, symbol))
        if "text" in data_types:
            tasks.append(self.search_vector_db(query, k_vector))
        if "graph" in data_types:
            tasks.append(self.search_knowledge_graph(query))

        results = await asyncio.gather(*tasks)
        
        for res in results:
            if isinstance(res, list):
                evidence_list.extend(res)
            else:
                evidence_list.append(res)

        return evidence_list

    # 修复：添加 worker.py 中使用的 (但 ai/retriever.py 中没有的) 方法
    
    async def retrieve_relevant_context(self, state: PipelineState, k: int) -> List[Dict[str, Any]]:
        """
        (旧版方法，可能被 reasoning_ensemble.py 调用)
        从向量存储中检索与当前状态相关的上下文。
        """
        log.warning("Using DEPRECATED method 'retrieve_relevant_context'. Should use 'retrieve_evidence'.")
        
        # 尝试从状态中获取一个查询
        query = state.get_main_task_query().get("description", "Analyze current market state")
        
        text_evidence = await self.search_vector_db(query, k=k)
        
        # 转换为旧版 reasoning_ensemble.py 可能期望的字典格式
        return [
            {
                "text": ev.text,
                "metadata": ev.metadata,
                "score": ev.score
            } for ev in text_evidence
        ]

    def format_context_for_prompt(self, context_docs: List[Dict[str, Any]]) -> str:
        """
        (旧版方法，可能被 reasoning_ensemble.py 调用)
        将检索到的文档格式化为 LLM 提示字符串。
        """
        log.warning("Using DEPRECATED method 'format_context_for_prompt'.")
        
        prompt_context = "--- CONTEXT DOCUMENTS ---\n\n"
        for i, doc in enumerate(context_docs):
            prompt_context += f"DOCUMENT {i+1} (Source: {doc.get('metadata', {}).get('source', 'N/A')}, Score: {doc.get('score', 0.0):.4f}):\n"
            prompt_context += doc.get('text', '')
            prompt_context += "\n\n-----------------\n"
        return prompt_context

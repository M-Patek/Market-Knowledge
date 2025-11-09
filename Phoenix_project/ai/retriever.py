import asyncio
import time # 修复：导入 time
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder # type: ignore
# 修复：导入正确的 monitor.logging 和 core.exceptions
from Phoenix_project.monitor.logging import get_logger
from Phoenix_project.memory.vector_store import VectorStore
# 修复：导入 ai.graph_db_client, ai.tabular_db_client, ai.temporal_db_client
from Phoenix_project.ai.graph_db_client import GraphDBClient
from Phoenix_project.ai.tabular_db_client import TabularDBClient
from Phoenix_project.ai.temporal_db_client import TemporalDBClient
from Phoenix_project.core.exceptions import RetrievalError

# 修复：使用 get_logger
logger = get_logger(__name__)

class Retriever:
    """
    从多个数据源（向量、图形、表格、时间序列）检索信息。
    """
    def __init__(
        self,
        vector_store: VectorStore,
        graph_db: GraphDBClient,
        tabular_db: TabularDBClient,
        temporal_db: TemporalDBClient,
        cross_encoder_model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        # 修复：添加在 worker.py 中传递但 __init__ 中缺失的依赖项
        config_loader: Any,
        embedding_client: Any,
        knowledge_graph_service: Any,
        cot_database: Any
    ):
        self.vector_store = vector_store
        self.graph_db = graph_db
        self.tabular_db = tabular_db
        self.temporal_db = temporal_db
        
        # 修复：存储缺失的依赖项
        self.config_loader = config_loader
        self.embedding_client = embedding_client
        self.knowledge_graph_service = knowledge_graph_service
        self.cot_database = cot_database
        
        # 修复：添加模型缓存
        self._reranker_cache: Dict[str, CrossEncoder] = {}
        self.reranker = self._load_reranker(cross_encoder_model_name)
        logger.info("Retriever initialized with all data sources.")

    def _load_reranker(self, model_name: str) -> Optional[CrossEncoder]:
        """
        加载 CrossEncoder reranker 模型。
        修复：为模型加载添加重试逻辑和缓存。
        """
        # 1. 检查缓存
        if model_name in self._reranker_cache:
            logger.debug(f"Loading CrossEncoder '{model_name}' from cache.")
            return self._reranker_cache[model_name]

        # 2. 添加重试逻辑
        max_retries = 3
        delay_seconds = 2
        for attempt in range(max_retries):
            try:
                model = CrossEncoder(model_name)
                logger.info(f"CrossEncoder model '{model_name}' loaded successfully (attempt {attempt + 1}).")
                # 3. 存入缓存
                self._reranker_cache[model_name] = model
                return model
            except Exception as e:
                logger.warning(f"Failed to load CrossEncoder model '{model_name}' (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay_seconds)
                else:
                    logger.error(f"Failed to load CrossEncoder model '{model_name}' after {max_retries} attempts.")
                    return None
        return None # 理论上不会执行到这里

    async def retrieve(
        self, 
        query: str, 
        top_k: int = 10, 
        tickers: Optional[List[str]] = None,
        data_types: Optional[List[str]] = None
    ) -> List[Document]:
        """
        从所有相关源异步检索和重排文档。
        
        Args:
            query: 用户的自然语言查询。
            top_k: 最终返回的文档数量。
            tickers: 资产负债表（例如 ["AAPL", "GOOG"]）。
            data_types: 要查询的数据源类型 (例如 ["news", "financials"])。

        Returns:
            一个经过重排和去重的文档列表。
        """
        if data_types is None:
            data_types = ["vector", "graph", "financials", "market_data"]
        
        logger.info(f"Starting retrieval for query: '{query}' with data_types: {data_types}")

        tasks = []
        if "vector" in data_types:
            tasks.append(self.search_vector_store(query, k=top_k * 2))
        
        if "graph" in data_types and tickers:
            tasks.append(self.search_graph_db(tickers))
            
        if "financials" in data_types and tickers:
            tasks.append(self.search_financials(tickers, query))
            
        if "market_data" in data_types and tickers:
            tasks.append(self.search_temporal_db(tickers, query))

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error during parallel retrieval: {e}")
            raise RetrievalError(f"Parallel retrieval failed: {e}")

        all_documents = []
        for res in results:
            if isinstance(res, Exception):
                logger.warning(f"Retrieval task failed: {res}")
            elif res:
                all_documents.extend(res)

        logger.info(f"Retrieved {len(all_documents)} raw documents before reranking.")

        # 去重和重排
        unique_documents = self._deduplicate(all_documents)
        reranked_documents = self.rerank(query, unique_documents, top_k=top_k)

        logger.info(f"Returning {len(reranked_documents)} reranked documents.")
        return reranked_documents

    async def search_vector_store(self, query: str, k: int) -> List[Document]:
        """
        从向量存储中检索文档。
        """
        try:
            logger.debug(f"Querying Vector Store: {query}")
            # 修复：VectorStore.similarity_search 返回 List[Tuple[Document, float]]
            results_with_scores = await self.vector_store.asimilarity_search(query, k=k)
            return [doc for doc, score in results_with_scores] # 仅提取文档
        except Exception as e:
            logger.error(f"Vector store search failed: {e}")
            return []

    async def search_graph_db(self, tickers: List[str]) -> List[Document]:
        """
        从图数据库中检索实体和关系。
        """
        try:
            logger.debug(f"Querying Graph DB for tickers: {tickers}")
            # 修复：GraphDBClient 没有 get_subgraph_for_tickers
            # 假设我们执行一个查询
            query = """
            MATCH (n) WHERE n.id IN $tickers
            OPTIONAL MATCH (n)-[r]-(m)
            RETURN n, r, m
            LIMIT 20
            """
            graph_data = await self.graph_db.execute_query(query, params={"tickers": tickers})
            if not graph_data:
                return []
            
            # 将图数据转换为 Document 对象
            content = f"Graph data for {', '.join(tickers)}: {graph_data}"
            return [Document(page_content=content, metadata={"source": "graph_db", "tickers": tickers})]
        except Exception as e:
            logger.error(f"Graph DB search failed: {e}")
            return []

    async def search_financials(self, tickers: List[str], query: str) -> List[Document]:
        """
        从表格数据库中检索财务数据。
        """
        all_results = []
        try:
            # 修复：调用 search_financials
            results_with_scores = await self.tabular_db.search_financials(query=query, symbols=tickers, limit=5 * len(tickers))
            
            for res_dict, score in results_with_scores:
                all_results.append(Document(
                    page_content=str(res_dict),
                    metadata={"source": "financials_db", "ticker": res_dict.get("symbol"), "score": score}
                ))
        except Exception as e:
            logger.warning(f"Failed to query tabular data for {tickers}: {e}")
                
        return all_results

    async def search_temporal_db(self, tickers: List[str], query: str) -> List[Document]:
        """
        从时间序列数据库检索市场数据。
        """
        try:
            logger.debug(f"Querying Temporal DB for tickers: {tickers} with query '{query}'")
            # 修复：调用 search_events
            results_with_scores = await self.temporal_db.search_events(
                symbols=tickers,
                query_string=query,
                size=10
            )
            docs = []
            for event_data, score in results_with_scores:
                content = f"Event: {event_data.get('headline', 'N/A')}\nSummary: {event_data.get('summary', 'N/A')}"
                docs.append(Document(page_content=content, metadata={"source": "temporal_db", "ticker": event_data.get("symbols"), "score": score}))
            return docs
        except Exception as e:
            logger.error(f"Temporal DB search failed: {e}")
            return []

    def rerank(self, query: str, documents: List[Document], top_k: int) -> List[Document]:
        """
        使用 CrossEncoder 模型对检索到的文档进行重排。
        """
        if not self.reranker:
            logger.warning("Reranker model not loaded. Returning top_k documents without reranking.")
            return documents[:top_k]
            
        if not documents:
            return []

        try:
            logger.debug(f"Reranking {len(documents)} documents for query: {query}")
            pairs = [(query, doc.page_content) for doc in documents]
            scores = self.reranker.predict(pairs) # type: ignore
            
            # 将分数与文档配对并排序
            scored_docs = list(zip(scores, documents))
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            
            # 返回 top_k 文档
            return [doc for score, doc in scored_docs[:top_k]]
            
        except Exception as e:
            return self._fallback_reranker(query, documents, top_k, e)

    def _fallback_reranker(self, query: str, documents: List[Document], top_k: int, e: Exception) -> List[Document]:
        """
        在 CrossEncoder 失败时使用的基于关键字的回退重排器。
        """
        # 如果 CrossEncoder 模型加载失败，则回退到基于关键字的重排器。
        # 未来的增强可以实现更健壮的本地模型加载或重试逻辑。
        logger.warning(f"CrossEncoder reranking failed: {e}. Using fallback keyword reranker.")
        
        # 简单的基于关键字的重排器作为回退
        query_terms = set(query.lower().split())
        
        def keyword_score(doc: Document) -> int:
            content = doc.page_content.lower()
            return sum(1 for term in query_terms if term in content)
            
        scored_docs = [(keyword_score(doc), doc) for doc in documents]
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        return [doc for score, doc in scored_docs[:top_k]]

    def _deduplicate(self, documents: List[Document]) -> List[Document]:
        """
        基于页面内容对文档进行去重。
        """
        seen_content = set()
        unique_docs = []
        for doc in documents:
            if doc.page_content not in seen_content:
                unique_docs.append(doc)
                seen_content.add(doc.page_content)
        return unique_docs

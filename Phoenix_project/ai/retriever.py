import asyncio
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder # type: ignore
from monitor.logging import logger
from memory.vector_store import VectorStore
from ai.graph_db_client import GraphDBClient
from ai.tabular_db_client import TabularDBClient
from ai.temporal_db_client import TemporalDBClient
from core.exceptions import RetrievalError

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
        cross_encoder_model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
    ):
        self.vector_store = vector_store
        self.graph_db = graph_db
        self.tabular_db = tabular_db
        self.temporal_db = temporal_db
        self.reranker = self._load_reranker(cross_encoder_model_name)
        logger.info("Retriever initialized with all data sources.")

    def _load_reranker(self, model_name: str) -> Optional[CrossEncoder]:
        """
        加载 CrossEncoder reranker 模型。
        """
        try:
            # TODO: Add model loading retry logic and caching
            model = CrossEncoder(model_name)
            logger.info(f"CrossEncoder model '{model_name}' loaded successfully.")
            return model
        except Exception as e:
            logger.warning(f"Failed to load CrossEncoder model '{model_name}': {e}. Reranking will be disabled.")
            return None

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
            return await self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"Vector store search failed: {e}")
            return []

    async def search_graph_db(self, tickers: List[str]) -> List[Document]:
        """
        从图数据库中检索实体和关系。
        """
        try:
            logger.debug(f"Querying Graph DB for tickers: {tickers}")
            graph_data = await self.graph_db.get_subgraph_for_tickers(tickers)
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
        # FIXME: 简化的搜索。更高级的系统会使用 query_to_sql_agent
        # 将自然语言 'query' 转换为 SQL 查询。
        # 目前，我们将为给定的 tickers 抓取最近的数据。
        
        all_results = []
        for ticker in tickers:
            try:
                # 假设一个表结构，并且 'query' 是一个提示，
                # 但我们只是在抓取常规数据。
                # 一个真正的实现会解析 'query' 来获取指标。
                query_params = {"ticker": ticker, "limit": 5} # 抓取 5 个最近的条目
                # 假设一个表名，例如 'financial_metrics'
                results = await self.tabular_db.query("financial_metrics", query_params)
                
                for res in results:
                    all_results.append(Document(
                        page_content=str(res),
                        metadata={"source": "financials_db", "ticker": ticker}
                    ))
            except Exception as e:
                logger.warning(f"Failed to query tabular data for {ticker}: {e}")
                
        return all_results

    async def search_temporal_db(self, tickers: List[str], query: str) -> List[Document]:
        """
        从时间序列数据库检索市场数据。
        """
        try:
            logger.debug(f"Querying Temporal DB for tickers: {tickers}")
            # FIXME: 'query' 没有被用来确定时间范围或指标
            # 简化：获取过去7天的数据
            results = await self.temporal_db.get_recent_data(tickers, days=7)
            docs = []
            for ticker, data in results.items():
                if not data.empty:
                    content = f"Recent market data for {ticker}:\n{data.to_string()}"
                    docs.append(Document(page_content=content, metadata={"source": "temporal_db", "ticker": ticker}))
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

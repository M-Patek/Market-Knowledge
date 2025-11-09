import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import CrossEncoder
from phoenix_project.ai.embedding_client import EmbeddingClient
from phoenix_project.ai.graph_db_client import GraphDBClient
from phoenix_project.ai.tabular_db_client import TabularDBClient
from phoenix_project.ai.temporal_db_client import TemporalDBClient
from phoenix_project.core.schemas.evidence_schema import (
    Evidence,
    FinancialsEvidence,
    TemporalEvidence,
    GraphEvidence,
    TextEvidence,
)
from phoenix_project.memory.vector_store import VectorStore
from phoenix_project.monitor.logging import get_logger

log = get_logger("Retriever")


class Retriever:
    """
    数据检索器，负责从所有数据源（向量、时序、表格、图）中检索信息。
    """

    def __init__(
        self,
        vector_store: VectorStore,
        temporal_db: TemporalDBClient,
        tabular_db: TabularDBClient,
        graph_db: GraphDBClient,
        embedding_client: EmbeddingClient,
        config: Dict[str, Any],
    ):
        self.vector_store = vector_store
        self.temporal_db = temporal_db
        self.tabular_db = tabular_db
        self.graph_db = graph_db
        self.embedding_client = embedding_client
        self.config = config
        self.reranker_model_name = self.config.get(
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

    def search_financials(self, query: str, symbol: str) -> FinancialsEvidence:
        """
        检索表格财务数据。
        [✅ 优化] 此处逻辑已移至 tabular_db_client，由其决定使用 SQL 代理还是 ILIKE。
        """
        try:
            results = self.tabular_db.search_financials(query, symbol)
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

    def search_temporal_db(self, query: str, symbol: str) -> TemporalEvidence:
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
            df = self.temporal_db.search_range(
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

    def search_vector_db(self, query: str, k: int) -> List[TextEvidence]:
        """
        从向量存储中检索非结构化文本。
        """
        try:
            query_embedding = self.embedding_client.embed(query)
            search_results = self.vector_store.search(query_embedding, k=k)

            # 重新排序 (Reranking)
            if self.reranker:
                passages = [
                    result.text for result in search_results
                ]
                # [query, passage] pairs
                rerank_input = [
                    (query, passage) for passage in passages
                ]
                scores = self.reranker.predict(rerank_input)

                # 组合结果和分数
                reranked_results = [
                    (search_results[i], scores[i]) for i in range(len(search_results))
                ]
                # 按新分数排序
                reranked_results.sort(key=lambda x: x[1], reverse=True)
                
                # 提取排序后的结果
                search_results = [res[0] for res in reranked_results]
                log.debug(f"Reranked {len(search_results)} results for query: '{query}'")

            return [
                TextEvidence(
                    text=result.text,
                    source=result.source,
                    timestamp=result.timestamp,
                    metadata=result.metadata,
                    query=query,
                    score=result.score,
                )
                for result in search_results
            ]
        except Exception as e:
            log.error(f"Error searching vector DB with query '{query}': {e}")
            return [TextEvidence(text="", source="VectorStore", query=query, error=str(e))]

    def search_knowledge_graph(self, query: str) -> GraphEvidence:
        """
        检索知识图谱。
        """
        try:
            # TODO: 实现一个 "text-to-Cypher" 的转换器
            cypher_query = f"MATCH (n) WHERE n.name CONTAINS '{query}' RETURN n"
            results = self.graph_db.query(cypher_query)
            return GraphEvidence(
                data=results, query=query, cypher_query=cypher_query, source="GraphDB"
            )
        except Exception as e:
            log.error(f"Error searching graph DB with query '{query}': {e}")
            return GraphEvidence(data=[], query=query, source="GraphDB", error=str(e))

    def retrieve_evidence(
        self, query: str, symbol: str, data_types: List[str], k_vector: int
    ) -> List[Evidence]:
        """
        根据数据类型检索所有相关证据。
        """
        evidence_list = []
        if "financials" in data_types:
            evidence_list.append(self.search_financials(query, symbol))
        if "temporal" in data_types:
            evidence_list.append(self.search_temporal_db(query, symbol))
        if "text" in data_types:
            evidence_list.extend(self.search_vector_db(query, k_vector))
        if "graph" in data_types:
            evidence_list.append(self.search_knowledge_graph(query))

        return evidence_list

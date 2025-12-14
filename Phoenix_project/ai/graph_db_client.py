import os
import asyncio
from neo4j import AsyncGraphDatabase
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator
from collections import defaultdict
from uuid import UUID
import re
import time

from Phoenix_project.monitor.logging import get_logger
from Phoenix_project.utils.retry import retry_with_exponential_backoff

logger = get_logger(__name__)

class GraphDBClient:
    """
    一个异步 Neo4j 数据库客户端。
    [Fix] 实现了通用的三元组写入逻辑，强制关系存储为边，防止图谱降维。
    [Task 2.3] Timestamp Fix: Removed internal timestamp() calls for Time Machine compatibility.
    """
    
    # 定义常规属性白名单 (大小写不敏感处理)
    # 这些谓词将被存储为节点属性，其他所有谓词将被存储为关系
    PROPERTY_KEYS = {
        'headline', 'content', 'summary', 'text', 'description',
        'timestamp', 'date', 'url', 'source', 'author',
        'sentiment', 'score', 'confidence', 'severity', 'impact', 'risk_level',
        'price', 'volume', 'market_cap', 'pe_ratio',
        'id', 'uid', 'ingestion_batch_id', 'chunk_index'
    }

    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = {}
        
        self.uri = config.get("uri_env") or os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
        self.user = config.get("user_env") or os.environ.get("NEO4J_USER", "neo4j")
        self.password = config.get("pass_env") or os.environ.get("NEO4J_PASSWORD", "password")
        
        try:
            logger.info(f"GraphDBClient: 正在初始化驱动程序，目标 {self.uri}")
            self.driver = AsyncGraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.log_prefix = "GraphDBClient:"
            logger.info(f"{self.log_prefix} 驱动程序已初始化。")

        except Exception as e:
            logger.error(f"{self.log_prefix} 初始化 Neo4j 驱动程序失败: {e}", exc_info=True)
            self.driver = None

    @retry_with_exponential_backoff()
    async def verify_connectivity(self):
        if not self.driver: return False
        try:
            await self.driver.verify_connectivity()
            return True
        except Exception:
            return False

    @retry_with_exponential_backoff()
    async def execute_query_stream(self, query: str, params: Optional[Dict[str, Any]] = None) -> AsyncGenerator[Dict[str, Any], None]:
        if not self.driver: return
        params = params or {}
        try:
            async with self.driver.session() as session:
                result = await session.run(query, params)
                async for record in result:
                    yield record.data()
        except Exception as e:
            logger.error(f"{self.log_prefix} Stream query failed: {e}", exc_info=True)
            raise

    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        data = []
        try:
            async for record in self.execute_query_stream(query, params):
                data.append(record)
            return data
        except Exception:
            return []

    @retry_with_exponential_backoff()
    async def execute_write(self, query: str, params: Optional[Dict[str, Any]] = None) -> bool:
        if not self.driver: return False
        params = params or {}
        async def _run_transaction(tx, q, p):
            await tx.run(q, p)
        try:
            async with self.driver.session() as session:
                await session.write_transaction(_run_transaction, query, params)
            return True
        except Exception as e:
            logger.error(f"{self.log_prefix} Cypher write failed: {e}", exc_info=True)
            return False

    async def ensure_schema_constraints(self):
        if not self.driver: return False
        # 约束：Symbol 节点的 'id' 必须是唯一的
        constraint_query = "CREATE CONSTRAINT symbol_id_unique IF NOT EXISTS FOR (s:Symbol) REQUIRE s.id IS UNIQUE"
        try:
            await self.execute_write(constraint_query)
            return True
        except Exception as e:
            logger.warning(f"{self.log_prefix} Schema constraint warning: {e}")
            return False
            
    def validate_cypher_query(self, query: str) -> bool:
        """
        验证 Cypher 查询的安全性与格式。
        目前仅做基础非空检查，未来可扩展 AST 校验。
        """
        if not query or not isinstance(query, str) or not query.strip():
            logger.warning(f"{self.log_prefix} Invalid or empty Cypher query.")
            return False
        # [Future] Add safety checks for DELETE/DETACH if read-only is required
        return True

    def _parse_id(self, entity_id: str) -> Tuple[str, str]:
        """
        Helper: 从 ID (Label:Value) 中解析 Label 和 UID。
        默认 Label 为 'Entity'。
        """
        if ':' in entity_id:
            parts = entity_id.split(':', 1)
            return self._sanitize_identifier(parts[0]), parts[1]
        return "Entity", entity_id

    def _sanitize_identifier(self, identifier: str) -> str:
        """
        Helper: 简单的标识符清洗，防止极端字符破坏 Cypher。
        保留字母、数字、下划线。
        """
        return re.sub(r'[^a-zA-Z0-9_]', '', identifier)

    @retry_with_exponential_backoff()
    async def add_triples(self, triples: List[Tuple[str, str, Any]], batch_id: Optional[UUID] = None, timestamp_ms: Optional[int] = None) -> bool:
        """
        [Phase 2 Task 3] 通用图谱写入。
        强制将非属性谓词写入为边 (Edge)。
        使用参数化查询和反引号转义防止注入。
        
        Args:
            timestamp_ms: Explicit timestamp in milliseconds. If None, uses current system time (Live mode only).
        """
        if not self.driver or not triples:
            return True
        
        # [Task 2.3] Explicit Timestamp Handling
        if timestamp_ms is None:
             timestamp_ms = int(time.time() * 1000)

        # 1. Bucket Data
        # node_props: Label -> NodeID -> {prop_key: prop_val}
        node_props = defaultdict(lambda: defaultdict(dict))
        
        # edges: (RelType, SourceLabel, TargetLabel) -> List[{s_id, t_id, props}]
        # Grouping by Labels allow us to use specific MERGE (n:Label) which is more performant
        edges = defaultdict(list)

        for s, p, o in triples:
            s_label, s_uid = self._parse_id(s)
            
            # Determine if Property or Edge
            is_property = p.lower() in self.PROPERTY_KEYS
            
            if is_property:
                # Store as Node Property
                # Ensure node exists in our tracking
                if s not in node_props[s_label]: 
                    node_props[s_label][s] = {'id': s, 'uid': s_uid}
                node_props[s_label][s][p] = o
            else:
                # Store as Relationship (Edge)
                # s -[p]-> o
                if not isinstance(o, str):
                    # Fallback: complex object as property despite predicate name
                     if s not in node_props[s_label]: node_props[s_label][s] = {'id': s, 'uid': s_uid}
                     node_props[s_label][s][p] = str(o)
                     continue

                t_label, t_uid = self._parse_id(o)
                
                # Sanitize RelType (p) -> Upper Snake Case
                rel_type = self._sanitize_identifier(p).upper()
                if not rel_type: rel_type = "RELATED_TO"
                
                key = (rel_type, s_label, t_label)
                edges[key].append({'s': s, 't': o})
                
                # Ensure endpoints exist in node tracking (minimal entry)
                if s not in node_props[s_label]: node_props[s_label][s] = {'id': s, 'uid': s_uid}
                if o not in node_props[t_label]: node_props[t_label][o] = {'id': o, 'uid': t_uid}

        try:
            batch_id_str = str(batch_id) if batch_id else None

            # 2. Execute Node Batch Writes (Per Label)
            for label, nodes_dict in node_props.items():
                batch_data = list(nodes_dict.values())
                # Use backticks for label safety
                # [Task 2.3] Use $ts parameter instead of timestamp()
                query = (
                    f"UNWIND $batch as item "
                    f"MERGE (n:`{label}` {{id: item.id}}) "
                    f"ON CREATE SET n.ingestion_batch_id = $batch_id, n.created_at = $ts "
                    f"SET n += item " # item contains props
                )
                await self.execute_write(query, {'batch': batch_data, 'batch_id': batch_id_str, 'ts': timestamp_ms})

            # 3. Execute Edge Batch Writes (Per RelType & Label Pair)
            for (rel_type, s_label, t_label), rels_list in edges.items():
                # Use backticks for labels and rel_type safety
                # [Task 2.3] Use $ts parameter
                query = (
                    f"UNWIND $batch as r "
                    f"MATCH (s:`{s_label}` {{id: r.s}}) "
                    f"MERGE (t:`{t_label}` {{id: r.t}}) " # Use MERGE for target to be safe, though MATCH is faster if strictly ordered
                    f"MERGE (s)-[rel:`{rel_type}`]->(t) "
                    f"ON CREATE SET rel.ingestion_batch_id = $batch_id, rel.created_at = $ts "
                )
                await self.execute_write(query, {'batch': rels_list, 'batch_id': batch_id_str, 'ts': timestamp_ms})

            logger.info(f"{self.log_prefix} Processed {len(triples)} triples (Nodes: {sum(len(v) for v in node_props.values())}, Edges: {sum(len(v) for v in edges.values())})")
            return True

        except Exception as e:
            logger.error(f"{self.log_prefix} add_triples failed: {e}", exc_info=True)
            return False

    async def count_by_batch_id(self, batch_id: UUID) -> int:
        if not self.driver: return 0
        query = "MATCH (n {ingestion_batch_id: $batch_id}) RETURN count(n) as count"
        try:
            results = await self.execute_query(query, {"batch_id": str(batch_id)})
            return results[0]["count"] if results else 0
        except Exception:
            return 0

    async def close(self):
        if self.driver:
            await self.driver.close()

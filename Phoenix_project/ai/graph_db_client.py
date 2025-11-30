import os
import asyncio
from neo4j import AsyncGraphDatabase
from typing import List, Dict, Any, Optional, Tuple, AsyncGenerator
from collections import defaultdict
from uuid import UUID

# 导入 Phoenix 项目的日志记录器
from Phoenix_project.monitor.logging import get_logger
from ..utils.retry import retry_with_exponential_backoff

logger = get_logger(__name__)

class GraphDBClient:
    """
    一个异步 Neo4j 数据库客户端。
    它从环境变量中读取连接详细信息，
    这些变量在 docker-compose.yml 中定义。
    """
    
    def __init__(self):
        """
        初始化 Neo4j 驱动程序。
        """
        try:
            # 从环境变量获取连接信息
            # 回退到 docker-compose.yml 中的默认值
            self.uri = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
            self.user = os.environ.get("NEO4J_USER", "neo4j")
            self.password = os.environ.get("NEO4J_PASSWORD", "password") # 来自 docker-compose
            
            logger.info(f"GraphDBClient: 正在初始化驱动程序，目标 {self.uri}")
            
            # 使用异步驱动程序
            self.driver = AsyncGraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.log_prefix = "GraphDBClient:"
            logger.info(f"{self.log_prefix} 驱动程序已初始化。")

        except Exception as e:
            logger.error(f"{self.log_prefix} 初始化 Neo4j 驱动程序失败: {e}", exc_info=True)
            self.driver = None

    @retry_with_exponential_backoff()
    async def verify_connectivity(self):
        """检查与 Neo4j 的连接。"""
        if not self.driver:
            logger.error(f"{self.log_prefix} 无法验证连接：驱动程序未初始化。")
            return False
        try:
            await self.driver.verify_connectivity()
            logger.info(f"{self.log_prefix} Neo4j 连接已成功验证。")
            return True
        except Exception as e:
            logger.error(f"{self.log_prefix} Neo4j 连接验证失败: {e}", exc_info=True)
            return False

    @retry_with_exponential_backoff()
    async def execute_query_stream(self, query: str, params: Optional[Dict[str, Any]] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """
        [Task 3.1] Streams results from Neo4j (Async Generator).
        Prevents OOM on large datasets.
        """
        if not self.driver:
            logger.error(f"{self.log_prefix} Stream query failed: Driver not initialized.")
            return
            
        params = params or {}
        # logger.debug(f"{self.log_prefix} Streaming query: {query} with params {params}")
        
        try:
            async with self.driver.session() as session:
                result = await session.run(query, params)
                async for record in result:
                    yield record.data()
        except Exception as e:
            logger.error(f"{self.log_prefix} Stream query failed: {e}", exc_info=True)
            raise

    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Wrapper for backward compatibility. Collects stream into list.
        WARNING: Use execute_query_stream for large datasets to avoid OOM.
        """
        data = []
        try:
            async for record in self.execute_query_stream(query, params):
                data.append(record)
            return data
        except Exception:
            return []

    @retry_with_exponential_backoff()
    async def execute_write(self, query: str, params: Optional[Dict[str, Any]] = None) -> bool:
        """
        对 Neo4j 数据库执行一个写入事务 (WRITE query)。
        
        Args:
            query (str): 要执行的写入查询 (e.g., CREATE, MERGE, SET)。
            params (Optional[Dict[str, Any]]): 查询参数。

        Returns:
            bool: 事务是否成功。
        """
        if not self.driver:
            logger.error(f"{self.log_prefix} 写入失败：驱动程序未初始化。")
            return False
            
        params = params or {}
        logger.debug(f"{self.log_prefix} 正在执行写入: {query} with params {params}")
        
        async def _run_transaction(tx, q, p):
            await tx.run(q, p)

        try:
            async with self.driver.session() as session:
                # 在事务中执行写入
                await session.write_transaction(_run_transaction, query, params)
            logger.debug(f"{self.log_prefix} 写入事务成功。")
            return True
        except Exception as e:
            logger.error(f"{self.log_prefix} Cypher 写入事务失败: {e}", exc_info=True)
            return False

    async def ensure_schema_constraints(self):
        """
        [Task I.3] 确保图 schema 约束已创建。
        这是一个幂等操作 (idempotent)。
        """
        if not self.driver:
            logger.error(f"{self.log_prefix} 无法确保约束：驱动程序未初始化。")
            return False
        
        logger.info(f"{self.log_prefix} 正在确保 schema 约束...")
        
        # 约束：Symbol 节点的 'id' 必须是唯一的
        # 这对于 gnn_engine.py 中的 MERGE (s:Symbol {id: pred.symbol}) 至关重要
        constraint_query = """
        CREATE CONSTRAINT symbol_id_unique IF NOT EXISTS
        FOR (s:Symbol) REQUIRE s.id IS UNIQUE
        """
        
        try:
            await self.execute_write(constraint_query)
            logger.info(f"{self.log_prefix} 约束 'symbol_id_unique' 已确保。")
            return True # [Fix] 返回布尔值以保持一致性
        except Exception as e:
            # (在某些 Neo4j 版本/配置中，CREATE IF NOT EXISTS 仍可能在并发中失败或出错)
            logger.warning(f"{self.log_prefix} 确保约束时出错 (可能已存在): {e}")
            return False # [Fix] 返回布尔值以保持一致性

    @retry_with_exponential_backoff()
    async def add_triples(self, triples: List[Tuple[str, str, Any]], batch_id: Optional[UUID] = None) -> bool:
        """
        [Task 3.1] Batch add triples using standard Cypher (No APOC).
        Groups triples by type/label to optimize execution.
        """
        if not self.driver:
            logger.error(f"{self.log_prefix} 写入三元组失败：驱动程序未初始化。")
            return False
        
        if not triples:
            logger.warning(f"{self.log_prefix} add_triples 被调用，但没有三元组。")
            return True

        # 1. Pre-process and Bucket Triples
        nodes_data = defaultdict(lambda: defaultdict(dict)) # Label -> ID -> Props
        
        l1_rels = []      # isAnalysisOf
        l2_target = []    # targetsSymbol
        l2_basis = []     # basedOnAnalysis
        
        RESTRICTED_KEYS = {'id', 'uid', 'ingestion_batch_id'}

        for s, p, o in triples:
            # Determine Label from ID (e.g., "Analysis:123" -> "Analysis")
            label = s.split(':')[0] if ':' in s else 'Unknown'
            
            # Bucket Relations
            if p == 'isAnalysisOf':
                l1_rels.append({'s': s, 'o': o})
                if s not in nodes_data[label]: nodes_data[label][s] = {} # Ensure node creation
            elif p == 'targetsSymbol':
                l2_target.append({'s': s, 'o': o})
                if s not in nodes_data[label]: nodes_data[label][s] = {}
            elif p == 'basedOnAnalysis':
                l2_basis.append({'s': s, 'o': o})
                if s not in nodes_data[label]: nodes_data[label][s] = {}
            else:
                # Bucket Properties (Guard against restricted keys)
                if p not in RESTRICTED_KEYS:
                    nodes_data[label][s][p] = o
        
        try:
            # 2. Execute Node Creation & Property Setting (Per Label)
            for label, nodes in nodes_data.items():
                batch = [{'id': nid, 'props': props} for nid, props in nodes.items()]
                # Dynamic Label Injection (Safe: label derived from internal logic)
                q_nodes = (
                    f"UNWIND $batch as item "
                    f"MERGE (n:`{label}` {{id: item.id}}) "
                    f"ON CREATE SET n.ingestion_batch_id = $batch_id, n.uid = split(item.id, ':')[1] "
                    f"SET n += item.props"
                )
                await self.execute_write(q_nodes, {'batch': batch, 'batch_id': str(batch_id) if batch_id else None})

            # 3. Execute Relations (Per Type)
            if l1_rels:
                q_l1 = "UNWIND $batch as r MERGE (s {id: r.s}) MERGE (o:Symbol {id: r.o}) MERGE (s)-[:IS_ANALYSIS_OF {timestamp: timestamp()}]->(o)"
                await self.execute_write(q_l1, {'batch': l1_rels})
            
            if l2_target:
                q_l2t = "UNWIND $batch as r MERGE (s {id: r.s}) MERGE (o:Symbol {id: r.o}) MERGE (s)-[:TARGETS_SYMBOL {timestamp: timestamp()}]->(o)"
                await self.execute_write(q_l2t, {'batch': l2_target})
                
            if l2_basis:
                q_l2b = "UNWIND $batch as r MERGE (s {id: r.s}) MERGE (o:Analysis {id: r.o}) MERGE (s)-[:BASED_ON_ANALYSIS {timestamp: timestamp()}]->(o)"
                await self.execute_write(q_l2b, {'batch': l2_basis})

            logger.info(f"{self.log_prefix} Successfully processed {len(triples)} triples (Batch: {batch_id}).")
            return True

        except Exception as e:
            logger.error(f"{self.log_prefix} add_triples failed: {e}", exc_info=True)
            return False

    async def count_by_batch_id(self, batch_id: UUID) -> int:
        """
        [Task 3C] Count the number of nodes created in a specific ingestion batch.
        """
        if not self.driver:
            logger.warning(f"{self.log_prefix} count_by_batch_id called but driver not initialized.")
            return 0
        
        query = "MATCH (n {ingestion_batch_id: $batch_id}) RETURN count(n) as count"
        
        try:
            results = await self.execute_query(query, {"batch_id": str(batch_id)})
            if results:
                count = results[0].get("count", 0)
                logger.info(f"{self.log_prefix} Found {count} nodes for batch_id {batch_id}.")
                return count
            return 0
        except Exception as e:
            logger.error(f"{self.log_prefix} Failed to count by batch_id {batch_id}: {e}", exc_info=True)
            return 0

    async def close(self):
        """关闭 Neo4j 驱动程序连接。"""
        if self.driver:
            logger.info(f"{self.log_prefix} 正在关闭 Neo4j 驱动程序...")
            await self.driver.close()
            logger.info(f"{self.log_prefix} Neo4j 驱动程序已关闭。")

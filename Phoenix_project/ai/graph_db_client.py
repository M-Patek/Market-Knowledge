import os
import asyncio
import re
from neo4j import AsyncGraphDatabase
from typing import List, Dict, Any, Optional, Tuple # [Task 3] 导入 Tuple
from uuid import UUID

# 导入 Phoenix 项目的日志记录器
from Phoenix_project.monitor.logging import get_logger
from ..core.exceptions import QuerySafetyError
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

    @staticmethod
    def validate_cypher_query(query: str):
        """
        [Task 2A] 验证 Cypher 查询是否为只读。
        检查是否存在 CREATE, SET, DELETE, MERGE 关键字。
        
        Args:
            query (str): 要验证的 Cypher 查询语句。
            
        Raises:
            QuerySafetyError: 如果检测到任何写入/修改关键字。
        """
        # \b 确保我们匹配的是整个单词
        # re.IGNORECASE 使其不区分大小写
        forbidden_keywords = r'\b(CREATE|SET|DELETE|MERGE)\b'
        if re.search(forbidden_keywords, query, re.IGNORECASE):
            logger.warning(f"QuerySafetyError: 检测到潜在的不安全查询。Query: '{query}'")
            raise QuerySafetyError(f"Query validation failed: Detected forbidden write/modify operation.")
        
        logger.debug(f"Cypher query validated successfully.")

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
    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        对 Neo4j 数据库执行一个只读的 Cypher 查询。
        
        Args:
            query (str): 要执行的 Cypher 查询语句。
            params (Optional[Dict[str, Any]]): 查询参数。

        Returns:
            List[Dict[str, Any]]: 结果记录列表，每条记录是一个字典。
        """
        if not self.driver:
            logger.error(f"{self.log_prefix} 查询失败：驱动程序未初始化。")
            return []
            
        params = params or {}
        logger.debug(f"{self.log_prefix} 正在执行查询: {query} with params {params}")
        
        try:
            async with self.driver.session() as session:
                result = await session.run(query, params)
                # 将 Neo4j 记录转换为标准字典
                data = [record.data() async for record in result]
                logger.debug(f"{self.log_prefix} 查询返回 {len(data)} 条记录。")
                return data
        except Exception as e:
            logger.error(f"{self.log_prefix} Cypher 查询执行失败: {e}", exc_info=True)
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
        [Task 3] 批量添加三元组 (subject_id, predicate, object_literal_or_id) 到图。
        这使用 UNWIND 和 APOC 来高效处理。
        
        警告: 此实现依赖 Neo4j APOC 插件 (CALL apoc.do.case)。
               请确保您的 Neo4j 实例 (docker-compose.yml) 已安装 APOC。
        
        Args:
            triples (List[Tuple[str, str, Any]]): (subject_id, predicate, object) 的列表。
            batch_id (Optional[UUID]): [Task 3C] The atomic ingestion batch ID to stamp on new nodes.
        """
        if not self.driver:
            logger.error(f"{self.log_prefix} 写入三元组失败：驱动程序未初始化。")
            return False
        
        if not triples:
            logger.warning(f"{self.log_prefix} add_triples 被调用，但没有三元组。")
            return True

        # 将三元组转换为字典列表以便参数化
        params_list = [{"s": t[0], "p": t[1], "o": t[2]} for t in triples]
        
        # 这个 Cypher 查询遍历每个三元组：
        # 1. MERGE 主语节点 (s) (基于其 'id')
        # 2. CALL apoc.do.case 检查谓词 (p)：
        #    - 如果 p 是 'isAnalysisOf' (L1), 'targetsSymbol' (L2), 'basedOnAnalysis' (L2),
        #      则 MERGE 对象节点 (o) 并创建关系 (s)-[r]->(o)。
        #    - 否则 (默认): 将 'o' 视为 (s) 上的属性并设置它。
        query = """
        UNWIND $triples as t
        
        // 步骤 1: 确定主语标签。L1 是 Analysis, L2 是 FusionDecision
        // (我们假设 t.s 包含前缀, e.g., "Analysis:uuid")
        WITH t, split(t.s, ':')[0] AS s_label, split(t.s, ':')[1] AS s_id
        MERGE (s {id: t.s})
        // 在创建时设置标签
        ON CREATE SET s.id = t.s, s.uid = s_id, s_label_dynamic = s_label, s.ingestion_batch_id = $batch_id
        WITH s, s_label_dynamic, t
        CALL apoc.create.addLabels(s, [s_label_dynamic]) YIELD node AS s_labeled
        
        // 步骤 2: 处理对象 (关系 或 属性)
        CALL apoc.do.case([
            // --- L1 关系 ---
            t.p = 'isAnalysisOf', 
            'MERGE (o:Symbol {id: t.o}) MERGE (s_labeled)-[:IS_ANALYSIS_OF {timestamp: timestamp()}]->(o)',
            
            // --- L2 关系 ---
            t.p = 'targetsSymbol', 
            'MERGE (o:Symbol {id: t.o}) MERGE (s_labeled)-[:TARGETS_SYMBOL {timestamp: timestamp()}]->(o)',
            
            t.p = 'basedOnAnalysis', 
            'MERGE (o:Analysis {id: t.o}) MERGE (s_labeled)-[:BASED_ON_ANALYSIS {timestamp: timestamp()}]->(o)'
        ],
        // ELSE (默认): 将 'o' 视为 (s) 上的属性
        'SET s_labeled[t.p] = t.o',
        {s_labeled: s_labeled, t: t}) YIELD value
        
        RETURN count(t) as triples_processed
        """
        
        try:
            query_params = {
                "triples": params_list,
                "batch_id": str(batch_id) if batch_id else None
            }
            # [Fix] 此处应使用 execute_write 而不是 execute_query
            success = await self.execute_write(query, params=query_params)
            if success:
                logger.info(f"{self.log_prefix} 成功添加/更新了 {len(params_list)} 个三元组 (Batch ID: {batch_id})。")
                return True
            else:
                logger.error(f"{self.log_prefix} add_triples 事务失败。")
                return False
        except Exception as e:
            logger.error(f"{self.log_prefix} add_triples 事务失败 (是否安装了 APOC?): {e}", exc_info=True)
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

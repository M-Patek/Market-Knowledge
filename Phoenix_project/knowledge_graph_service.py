"""
知识图谱服务 (Knowledge Graph Service)
[已优化] 负责构建、更新和查询连接到 Neo4j 实例的知识图谱 (KG)。
"""
import asyncio
import os
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase, AsyncGraphDatabase
from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

class KnowledgeGraphService:
    """
    [已优化]
    管理知识图谱的生命周期，使用 neo4j 驱动程序连接到
    docker-compose.yml 中定义的 Neo4j 服务。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化 Neo4j 驱动程序。
        """
        try:
            # 从环境变量获取连接信息
            # 回退到 docker-compose.yml 中的默认值
            uri = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
            user = os.environ.get("NEO4J_USER", "neo4j")
            password = os.environ.get("NEO4J_PASSWORD", "password") # 来自 docker-compose
            
            logger.info(f"KnowledgeGraphService: 正在尝试连接到 Neo4j at {uri}")
            
            # 使用异步驱动程序
            self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
            
            # (在 asyncio 事件循环中，我们不能在这里同步验证连接)
            # (我们将在第一次查询时验证)
            self.log_prefix = "KnowledgeGraphService:"
            logger.info(f"{self.log_prefix} 驱动程序已初始化。")

        except Exception as e:
            logger.error(f"{self.log_prefix} 初始化 Neo4j 驱动程序失败: {e}", exc_info=True)
            self.driver = None

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

    async def query(self, cypher_query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        [已优化]
        对 Neo4j 数据库执行一个只读的 Cypher 查询。
        
        Args:
            cypher_query (str): 要执行的 Cypher 查询语句。
            params (Optional[Dict[str, Any]]): 查询参数。

        Returns:
            List[Dict[str, Any]]: 结果记录列表，每条记录是一个字典。
        """
        if not self.driver:
            logger.error(f"{self.log_prefix} 查询失败：驱动程序未初始化。")
            return []
            
        params = params or {}
        logger.debug(f"{self.log_prefix} 正在执行查询: {cypher_query} with params {params}")
        
        try:
            async with self.driver.session() as session:
                result = await session.run(cypher_query, params)
                # 将 Neo4j 记录转换为标准字典
                data = [record.data() async for record in result]
                logger.debug(f"{self.log_prefix} 查询返回 {len(data)} 条记录。")
                return data
        except Exception as e:
            logger.error(f"{self.log_prefix} Cypher 查询执行失败: {e}", exc_info=True)
            return []

    async def execute_write(self, cypher_query: str, params: Optional[Dict[str, Any]] = None) -> bool:
        """
        [新功能]
        对 Neo4j 数据库执行一个写入事务 (WRITE query)。
        供 KnowledgeInjector 使用。
        
        Args:
            cypher_query (str): 要执行的写入查询 (e.g., CREATE, MERGE, SET)。
            params (Optional[Dict[str, Any]]): 查询参数。

        Returns:
            bool: 事务是否成功。
        """
        if not self.driver:
            logger.error(f"{self.log_prefix} 写入失败：驱动程序未初始化。")
            return False
            
        params = params or {}
        logger.debug(f"{self.log_prefix} 正在执行写入: {cypher_query} with params {params}")
        
        try:
            async with self.driver.session() as session:
                # 在事务中执行写入
                await session.write_transaction(
                    lambda tx, q, p: tx.run(q, p), 
                    cypher_query, 
                    params
                )
            logger.debug(f"{self.log_prefix} 写入事务成功。")
            return True
        except Exception as e:
            logger.error(f"{self.log_prefix} Cypher 写入事务失败: {e}", exc_info=True)
            return False

    async def close(self):
        """关闭 Neo4j 驱动程序连接。"""
        if self.driver:
            logger.info(f"{self.log_prefix} 正在关闭 Neo4j 驱动程序...")
            await self.driver.close()
            logger.info(f"{self.log_prefix} Neo4j 驱动程序已关闭。")

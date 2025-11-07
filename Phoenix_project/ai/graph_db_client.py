import os
import asyncio
from neo4j import AsyncGraphDatabase
from typing import List, Dict, Any, Optional

# 导入 Phoenix 项目的日志记录器
from Phoenix_project.monitor.logging import get_logger

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

    async def close(self):
        """关闭 Neo4j 驱动程序连接。"""
        if self.driver:
            logger.info(f"{self.log_prefix} 正在关闭 Neo4j 驱动程序...")
            await self.driver.close()
            logger.info(f"{self.log_prefix} Neo4j 驱动程序已关闭。")

"""
知识图谱服务 (Knowledge Graph Service)
[蓝图 3 已更新] 
负责构建、更新和查询连接到 Neo4j 实例的知识图谱 (KG)。
这个实现替换了之前的 graph_db_stub 存根。
"""
import asyncio
from typing import List, Dict, Any, Optional

# 导入新的 GraphDBClient
from Phoenix_project.ai.graph_db_client import GraphDBClient
from Phoenix_project.monitor.logging import get_logger

logger = get_logger(__name__)

class KnowledgeGraphService:
    """
    [蓝图 3 已更新]
    管理知识图谱的生命周期，使用 GraphDBClient 连接到
    在 docker-compose.yml 中定义的 Neo4j 服务。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化 Neo4j 客户端。
        'config' 参数被保留以兼容旧的实例化方式，但 GraphDBClient
        会从环境变量中读取其配置。
        """
        self.db_client = GraphDBClient()
        self.log_prefix = "KnowledgeGraphService:"
        logger.info(f"{self.log_prefix} 已初始化，使用 GraphDBClient。")

    async def verify_connectivity(self):
        """检查与 Neo4j 的连接。"""
        return await self.db_client.verify_connectivity()

    async def close(self):
        """关闭 Neo4j 驱动程序连接。"""
        await self.db_client.close()

    async def query(self, cypher_query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        对 Neo4j 数据库执行一个只读的 Cypher 查询。
        """
        return await self.db_client.execute_query(cypher_query, params)

    async def update_knowledge_graph(self, extracted_data: Dict[str, List[Dict]]) -> bool:
        """
        [蓝图 3 新增]
        接收来自 RelationExtractor 的原始节点和边，
        并使用 Cypher MERGE 语句将它们写入 Neo4j。
        
        Args:
            extracted_data (Dict): 包含 "nodes" 和 "edges" 键的字典。
                - "nodes": [{"id": "AAPL", "type": "Company", "properties": {"name": "Apple Inc."}}]
                - "edges": [{"id": "rel_1", "source": "AAPL", "target": "TSMC", "type": "SUPPLIER_OF", "properties": {"product": "M3 Chip"}}]

        Returns:
            bool: 如果所有更新都成功，则为 True。
        """
        nodes = extracted_data.get("nodes", [])
        edges = extracted_data.get("edges", [])
        
        if not nodes and not edges:
            logger.warning(f"{self.log_prefix} update_knowledge_graph 接收到空数据，无需操作。")
            return True

        node_success = True
        edge_success = True

        # 1. 合并节点 (Nodes)
        # 我们假设节点 'type' 是标签 (Label)
        for node in nodes:
            try:
                node_id = node.get("id")
                node_label = node.get("type", "Node") # 回退到通用 "Node" 标签
                properties = node.get("properties", {})
                
                # 确保 id 在 properties 中，以便 SET n = $props 生效
                properties["id"] = node_id
                
                # 清理标签，防止 Cypher 注入
                if not node_label.isalnum():
                    logger.warning(f"无效的节点标签 '{node_label}'，将使用 'Node'。")
                    node_label = "Node"
                
                # MERGE (n:Label {id: $id}) ON CREATE SET n = $props ON MATCH SET n += $props
                # 'id' 是我们的唯一键
                # 'props' 包含了所有属性，包括 'id'
                query = f"""
                MERGE (n:{node_label} {{id: $id}})
                ON CREATE SET n = $props
                ON MATCH SET n += $props
                """
                
                params = {
                    "id": node_id,
                    "props": properties
                }
                
                if not await self.db_client.execute_write(query, params):
                    logger.error(f"{self.log_prefix} 写入节点 {node_id} 失败。")
                    node_success = False

            except Exception as e:
                logger.error(f"{self.log_prefix} 处理节点 {node.get('id')} 时出错: {e}", exc_info=True)
                node_success = False
        
        # 2. 合并边 (Edges)
        for edge in edges:
            try:
                edge_id = edge.get("id")
                source_id = edge.get("source")
                target_id = edge.get("target")
                edge_type = edge.get("type", "RELATED_TO") # 回退到通用关系
                properties = edge.get("properties", {})
                
                if not all([edge_id, source_id, target_id]):
                    logger.warning(f"{self.log_prefix} 边缺少 id, source 或 target，已跳过: {edge}")
                    continue

                # 清理关系类型
                if not edge_type.isalnum() or edge_type.upper() == "NODE":
                    logger.warning(f"无效的关系类型 '{edge_type}'，将使用 'RELATED_TO'。")
                    edge_type = "RELATED_TO"
                    
                properties["id"] = edge_id
                
                # 找到源节点和目标节点，然后合并关系
                # 我们假设源节点和目标节点使用 'id' 作为键
                query = f"""
                MATCH (a {{id: $source_id}}), (b {{id: $target_id}})
                MERGE (a)-[r:{edge_type} {{id: $edge_id}}]->(b)
                ON CREATE SET r = $props
                ON MATCH SET r += $props
                """
                
                params = {
                    "source_id": source_id,
                    "target_id": target_id,
                    "edge_id": edge_id,
                    "props": properties
                }
                
                if not await self.db_client.execute_write(query, params):
                    logger.error(f"{self.log_prefix} 写入边 {edge_id} 失败。")
                    edge_success = False

            except Exception as e:
                logger.error(f"{self.log_prefix} 处理边 {edge.get('id')} 时出错: {e}", exc_info=True)
                edge_success = False
                
        return node_success and edge_success

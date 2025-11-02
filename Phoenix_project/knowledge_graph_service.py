"""
Knowledge Graph Service

A high-level service that manages the Knowledge Graph (KG).
It provides an interface for:
1. Injecting new information (from RelationExtractor).
2. Querying the graph.
3. Potentially persisting the graph to a database (e.g., Neo4j, or
   simplified in PostgreSQL JSONB).

This service acts as an abstraction layer over the chosen KG backend.
"""
import logging
from typing import Optional, List

# 修复：从根目录开始使用绝对导入，移除 `.`
from ai.tabular_db_client import TabularDBClient # Example: Using PG for storage
from core.schemas.data_schema import KnowledgeGraph, KGNode, KGRelation

logger = logging.getLogger(__name__)

class KnowledgeGraphService:
    """
    Manages the creation, updating, and querying of the knowledge graph.
    
    This is a simplified implementation assuming storage in the
    existing TabularDB (PostgreSQL) using JSONB.
    """

    def __init__(self, db_client: TabularDBClient):
        """
        Initializes the KnowledgeGraphService.

        Args:
            db_client: An initialized TabularDBClient (PostgreSQL).
        """
        self.db = db_client
        logger.info("KnowledgeGraphService initialized (PostgreSQL backend).")

    async def setup_schema(self):
        """
        Ensures the tables for storing nodes and relations exist.
        """
        if not self.db.is_connected():
            logger.info("DB not connected, connecting for KG schema setup...")
            await self.db.connect()
            
        logger.info("Setting up Knowledge Graph schema...")
        
        # Table for Nodes (Entities)
        # We use ON CONFLICT to allow idempotent updates
        await self.db.execute("""
        CREATE TABLE IF NOT EXISTS kg_nodes (
            node_id VARCHAR(255) PRIMARY KEY,
            node_type VARCHAR(100) NOT NULL,
            name VARCHAR(255) NOT NULL,
            metadata JSONB,
            last_updated TIMESTAMPTZ NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_kg_node_type ON kg_nodes(node_type);
        CREATE INDEX IF NOT EXISTS idx_kg_node_name ON kg_nodes(name);
        """)
        
        # Table for Relations (Edges)
        await self.db.execute("""
        CREATE TABLE IF NOT EXISTS kg_relations (
            relation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            source_node_id VARCHAR(255) REFERENCES kg_nodes(node_id) ON DELETE CASCADE,
            target_node_id VARCHAR(255) REFERENCES kg_nodes(node_id) ON DELETE CASCADE,
            relation_type VARCHAR(100) NOT NULL,
            context TEXT,
            metadata JSONB,
            last_updated TIMESTAMPTZ NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_kg_rel_source ON kg_relations(source_node_id);
        CREATE INDEX IF NOT EXISTS idx_kg_rel_target ON kg_relations(target_node_id);
        CREATE INDEX IF NOT EXISTS idx_kg_rel_type ON kg_relations(relation_type);
        """)
        
        logger.info("Knowledge Graph schema setup complete.")

    async def add_knowledge_graph(self, kg: KnowledgeGraph):
        """
        Adds or updates nodes and relations from a KnowledgeGraph object
        into the database.
        """
        if not self.db.is_connected():
            logger.error("Database not connected. Cannot add KnowledgeGraph.")
            return

        now = "NOW()" # Use SQL NOW() for consistency

        async with self.db.pool.acquire() as conn:
            async with conn.transaction():
                try:
                    # 1. Upsert Nodes
                    for node in kg.nodes:
                        await conn.execute(
                            """
                            INSERT INTO kg_nodes (node_id, node_type, name, metadata, last_updated)
                            VALUES ($1, $2, $3, $4::jsonb, $5)
                            ON CONFLICT (node_id) DO UPDATE SET
                                name = EXCLUDED.name,
                                metadata = EXCLUDED.metadata,
                                last_updated = EXCLUDED.last_updated;
                            """,
                            node.id, node.type, node.name, node.metadata, now
                        )
                    
                    # 2. Insert Relations
                    # We assume relations are new and don't check for conflicts
                    # A more robust system would hash relations to prevent duplicates
                    for rel in kg.relations:
                        await conn.execute(
                            """
                            INSERT INTO kg_relations (source_node_id, target_node_id, relation_type, context, metadata, last_updated)
                            VALUES ($1, $2, $3, $4, $5::jsonb, $6);
                            """,
                            rel.source_id, rel.target_id, rel.type, rel.context, rel.metadata, now
                        )
                    
                    logger.info(f"Successfully upserted {len(kg.nodes)} nodes and added {len(kg.relations)} relations.")

                except Exception as e:
                    logger.error(f"Failed to add KnowledgeGraph (transaction rolled back): {e}", exc_info=True)
                    raise

    async def find_related_nodes(self, node_id: str, hops: int = 1) -> Optional[List[Dict[str, Any]]]:
        """
        Finds nodes connected to a given node_id within N hops.
        
        Args:
            node_id: The starting node ID (e.g., "AAPL").
            hops: The number of hops (1 = direct neighbors).

        Returns:
            A list of node and relation data, or None on failure.
        """
        if hops != 1:
            # Multi-hop queries are complex (require CTEs)
            logger.warning("Only 1-hop queries are supported in this simplified service.")
        
        query = """
        SELECT 
            'outbound' as direction, r.relation_type, t.*
        FROM kg_relations r
        JOIN kg_nodes t ON r.target_node_id = t.node_id
        WHERE r.source_node_id = $1
        
        UNION
        
        SELECT 
            'inbound' as direction, r.relation_type, s.*
        FROM kg_relations r
        JOIN kg_nodes s ON r.source_node_id = s.node_id
        WHERE r.target_node_id = $1;
        """
        try:
            results = await self.db.fetch(query, node_id)
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Error querying related nodes for {node_id}: {e}", exc_info=True)
            return None

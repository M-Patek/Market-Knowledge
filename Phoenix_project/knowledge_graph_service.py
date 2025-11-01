from neo4j import AsyncGraphDatabase
from typing import Dict, Any, List, Optional
from .monitor.logging import get_logger

logger = get_logger(__name__)

class KnowledgeGraphService:
    """
    Manages connections and queries to a Neo4j Knowledge Graph.
    Used for storing and retrieving complex relationships between
    entities (e.g., "Company A" -> "SUPPLIES" -> "Company B").
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the KnowledgeGraphService.

        Expected config keys:
          - uri: Neo4j bolt/bolt+s URI
          - user: Username for Neo4j
          - password: Password for Neo4j
          - database: Database name (optional; defaults to 'neo4j')
        """
        self.uri = config.get("uri")
        self.user = config.get("user")
        self.password = config.get("password")
        self.database = config.get("database", "neo4j")

        if not self.uri or not self.user or not self.password:
            raise ValueError("Neo4j config requires 'uri', 'user', and 'password'.")

        self._driver = AsyncGraphDatabase.driver(self.uri, auth=(self.user, self.password))
        logger.info("KnowledgeGraphService initialized for %s", self.uri)

    async def close(self):
        """Closes the underlying driver."""
        await self._driver.close()
        logger.info("KnowledgeGraphService closed.")

    async def run_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Runs a read-only query against the Neo4j database.
        """
        params = params or {}
        logger.debug("KG read query: %s | params=%s", query, params)
        async with self._driver.session(database=self.database) as session:
            result = await session.run(query, **params)
            records = [r.data() for r in await result.to_list()]
            logger.debug("KG read result: %s rows", len(records))
            return records

    async def run_write_tx(self, query: str, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Runs a write transaction with the given query/params.
        """
        params = params or {}
        logger.debug("KG write query: %s | params=%s", query, params)
        async with self._driver.session(database=self.database) as session:
            await session.execute_write(lambda tx: tx.run(query, **params))
        logger.debug("KG write commit ok.")

    async def upsert_entity(self, entity_id: str, entity_type: str, properties: Dict[str, Any]):
        """
        Creates or updates an entity node.
        """
        query = """
        MERGE (e:Entity {name: $entity_id})
        ON CREATE SET e.type = $entity_type
        SET e += $properties
        """
        await self.run_write_tx(
            query,
            {"entity_id": entity_id, "entity_type": entity_type, "properties": properties},
        )

    async def upsert_relation(self, entity_a_id: str, relation: str, entity_b_id: str, weight: float = 1.0):
        """
        Creates or updates a relation between two entities.
        """
        query = """
        MERGE (a:Entity {name: $entity_a})
        MERGE (b:Entity {name: $entity_b})
        MERGE (a)-[r:%s]->(b)
        ON CREATE SET r.weight = $weight
        SET r.weight = $weight
        """ % relation
        await self.run_write_tx(query, {"entity_a": entity_a_id, "entity_b": entity_b_id, "weight": weight})

    async def delete_entity(self, entity_id: str):
        """
        Deletes an entity and all relationships.
        """
        query = """
        MATCH (e:Entity {name: $entity_id})
        DETACH DELETE e
        """
        await self.run_write_tx(query, {"entity_id": entity_id})

    async def get_neighbors(self, entity_id: str, relation: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Returns neighbors connected with optional relation filter.
        """
        if relation:
            query = f"""
            MATCH (:Entity {{name: $entity_id}})-[r:{relation}]->(n:Entity)
            RETURN n.name AS neighbor, n.type AS type, r.weight AS weight
            """
        else:
            query = """
            MATCH (:Entity {name: $entity_id})-[r]->(n:Entity)
            RETURN n.name AS neighbor, n.type AS type, r.weight AS weight, type(r) AS relation
            """
        return await self.run_query(query, {"entity_id": entity_id})

    async def connect_entities(self, entity_a_id: str, entity_b_id: str):
        """
        Example helper to connect two entities with a generic relation.
        """
        query = """
        MATCH (a:Entity {name: $entity_a})
        MATCH (b:Entity {name: $entity_b})
        MERGE (a)-[:RELATED_TO]->(b)
        """
        await self.run_write_tx(query, {"entity_a": entity_a_id, "entity_b": entity_b_id})

    async def get_related_entities(self, entity_id: str, hops: int = 1) -> List[Dict]:
        """
        Example: Finds entities related to a given entity within N hops.
        """
        if hops < 1:
            hops = 1

        query = f"""
        MATCH (a:Entity {{name: $entity_id}})-[*1..{hops}]-(b:Entity)
        WHERE a <> b
        RETURN DISTINCT b.name AS related_entity, b.type AS entity_type
        """
        return await self.run_query(query, {"entity_id": entity_id})

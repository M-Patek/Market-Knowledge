from neo4j import AsyncGraphDatabase
from typing import Dict, Any, List, Optional
from ..monitor.logging import get_logger

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
        
        Args:
            config (Dict, Any): Expects 'knowledge_graph' config block
                                with 'uri', 'user', 'password'.
        """
        kg_config = config.get('knowledge_graph', {})
        self.uri = kg_config.get('uri') # e.g., "neo4j://localhost:7687"
        self.user = kg_config.get('user', 'neo4j')
        self.password = kg_config.get('password')
        
        if not self.uri or not self.password:
            logger.error("KnowledgeGraphService config (uri, password) incomplete.")
            raise ValueError("KG URI and password are required.")
            
        self.driver = None
        logger.info(f"KnowledgeGraphService initialized for URI: {self.uri}")

    async def connect(self):
        """Establishes the connection to the Neo4j driver."""
        try:
            self.driver = AsyncGraphDatabase.driver(self.uri, auth=(self.user, self.password))
            await self.driver.verify_connectivity()
            logger.info("KnowledgeGraphService connected to Neo4j.")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}", exc_info=True)
            raise

    async def close(self):
        """Closes the Neo4j driver connection."""
        if self.driver:
            await self.driver.close()
            logger.info("KnowledgeGraphService connection to Neo4j closed.")

    async def run_query(self, query: str, parameters: Dict = None) -> List[Dict]:
        """
        Executes a read-only Cypher query.
        
        Args:
            query (str): The Cypher query.
            parameters (Dict, optional): Parameters for the query.
            
        Returns:
            List[Dict]: A list of result records (as dictionaries).
        """
        if not self.driver:
            logger.error("Driver not connected.")
            return []
            
        try:
            async with self.driver.session() as session:
                result = await session.run(query, parameters)
                records = [record.data() async for record in result]
                return records
        except Exception as e:
            logger.error(f"Neo4j query failed: {e}. Query: {query}", exc_info=True)
            return []

    async def run_write_tx(self, query: str, parameters: Dict = None) -> bool:
        """
        Executes a write query within a transaction.
        
        Args:
            query (str): The Cypher write query (e.g., CREATE, MERGE).
            parameters (Dict, optional): Parameters for the query.
            
        Returns:
            bool: True on success, False on failure.
        """
        if not self.driver:
            logger.error("Driver not connected.")
            return False
            
        try:
            async with self.driver.session() as session:
                await session.write_transaction(self._execute_tx, query, parameters)
            return True
        except Exception as e:
            logger.error(f"Neo4j write transaction failed: {e}. Query: {query}", exc_info=True)
            return False

    @staticmethod
    async def _execute_tx(tx, query: str, parameters: Dict = None):
        """Internal helper for executing the transaction function."""
        await tx.run(query, parameters)

    # --- Example Specific Queries ---
    
    async def add_relationship(self, entity_a_id: str, entity_b_id: str, relationship: str):
        """
        Example: Creates a relationship between two entities.
        
        (Assumes nodes have a 'name' property as their ID)
        """
        query = """
        MERGE (a:Entity {name: $entity_a})
        MERGE (b:Entity {name: $entity_b})
        MERGE (a)-[r:""" + relationship.upper() + """]->(b)
        RETURN r
        """
        await self.run_write_tx(query, {"entity_a": entity_a_id, "entity_b": entity_b_id})

    async def get_related_entities(self, entity_id: str, hops: int = 1) -> List[Dict]:
        """
        Example: Finds entities related to a given entity within N hops.
        """
        if hops < 1: hops = 1
        
        query = f"""
        MATCH (a:Entity {{name: $entity_id}})-[*1..{hops}]-(b:Entity)
        WHERE a <> b
        RETURN DISTINCT b.name AS related_entity, b.type AS entity_type
        """
        return await self.run_query(query, {"entity_id": entity_id})

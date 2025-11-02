from typing import Dict, Any, List

from ai.tabular_db_client import TabularDatabaseClient
from ai.temporal_db_client import TemporalDatabaseClient
from monitor.logging import ESLogger


class KnowledgeGraphService:
    """
    Service responsible for interacting with and managing the knowledge graph.

    This service abstracts the complexities of querying and updating the
    temporal (e.g., KGraph) and tabular (e.g., BigQuery) databases that
    together form the knowledge graph.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        temporal_db: TemporalDatabaseClient,
        tabular_db: TabularDatabaseClient,
        logger: ESLogger,
    ):
        """
        Initializes the KnowledgeGraphService.

        Args:
            config: Configuration dictionary for the service.
            temporal_db: Client for the temporal database (e.g., KGraph).
            tabular_db: Client for the tabular database (e.g., BigQuery).
            logger: An instance of ESLogger for logging.
        """
        self.config = config
        self.temporal_db = temporal_db
        self.tabular_db = tabular_db
        self.logger = logger
        self.logger.log_info("KnowledgeGraphService initialized.")

    async def query_knowledge_graph(self, query: str, state: Any) -> Dict[str, Any]:
        """
        Queries the knowledge graph based on the current state and a specific query.

        This might involve:
        1. Decomposing the query.
        2. Querying the temporal DB for recent/fast-changing data.
        3. Querying the tabular DB for historical/structured data.
        4. Fusing the results.

        Args:
            query: The natural language or structured query.
            state: The current pipeline state, providing context (e.g., timestamps).

        Returns:
            A dictionary containing the fused results from the KG.
        """
        self.logger.log_debug(f"Querying Knowledge Graph with: {query}")
        try:
            # In a real implementation, query decomposition would happen here.
            # For this example, we'll query both and fuse naively.
            
            temporal_query = self._build_temporal_query(query, state)
            tabular_query = self._build_tabular_query(query, state)

            temporal_results = await self.temporal_db.query(temporal_query)
            tabular_results = await self.tabular_db.query(tabular_query)

            fused_results = self._fuse_results(temporal_results, tabular_results)
            
            self.logger.log_info(f"Knowledge Graph query successful for: {query}")
            return fused_results

        except Exception as e:
            self.logger.log_error(
                f"Error querying Knowledge Graph: {e}", exc_info=True
            )
            return {"error": str(e), "temporal_results": [], "tabular_results": []}

    async def update_knowledge_graph(self, data: Dict[str, Any]) -> bool:
        """
        Updates the knowledge graph with new information.

        This could involve writing to either the temporal or tabular DB,
        or both, depending on the nature of the data.

        Args:
            data: The data to be inserted or updated in the KG.
                  This data should be structured to indicate its destination
                  (e.g., {"temporal_updates": [...], "tabular_inserts": [...]}).

        Returns:
            True if all updates were successful, False otherwise.
        """
        self.logger.log_debug(f"Updating Knowledge Graph with data: {data}")
        success = True
        try:
            if "temporal_updates" in data:
                temporal_success = await self.temporal_db.update(
                    data["temporal_updates"]
                )
                if not temporal_success:
                    success = False
                    self.logger.log_warning("Temporal DB update failed.")

            if "tabular_inserts" in data:
                tabular_success = await self.tabular_db.insert(
                    data["tabular_inserts"]
                )
                if not tabular_success:
                    success = False
                    self.logger.log_warning("Tabular DB insert failed.")
            
            if success:
                self.logger.log_info("Knowledge Graph update successful.")
            else:
                self.logger.log_warning("Knowledge Graph update partially failed.")
                
            return success

        except Exception as e:
            self.logger.log_error(
                f"Error updating Knowledge Graph: {e}", exc_info=True
            )
            return False

    def _build_temporal_query(self, query: str, state: Any) -> str:
        """Builds a specific query for the temporal database."""
        # Example: Add time constraints from the state
        timestamp = state.get_timestamp()
        return f"FIND {query} AROUND {timestamp}"

    def _build_tabular_query(self, query: str, state: Any) -> str:
        """Builds a specific SQL query for the tabular database."""
        # Example: Use state to determine relevant tables
        market = state.get_market_context()
        return f"SELECT * FROM `{market}_data` WHERE CONTAINS(description, '{query}')"

    def _fuse_results(
        self, temporal_results: List[Any], tabular_results: List[Any]
    ) -> Dict[str, Any]:
        """Fuses results from both databases."""
        # This is a naive fusion. A real implementation would be more complex,
        # resolving entities and prioritizing recent data.
        return {
            "temporal_data": temporal_results,
            "historical_data": tabular_results,
            "fused_summary": f"Found {len(temporal_results)} recent items and {len(tabular_results)} historical items.",
        }

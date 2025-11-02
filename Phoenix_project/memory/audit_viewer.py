from typing import List, Dict, Any, Optional
from datetime import datetime
import json

from memory.cot_database import CoTDatabase
from monitor.logging import ESLogger


class AuditViewer:
    """
    Provides a query interface for the Chain-of-Thought (CoT) database
    to facilitate auditing, debugging, and transparency.
    """

    def __init__(self, cot_database: CoTDatabase, logger: ESLogger):
        """
        Initializes the AuditViewer.

        Args:
            cot_database: The CoTDatabase instance to query.
            logger: An instance of ESLogger for logging.
        """
        self.cot_database = cot_database
        self.logger = logger
        self.logger.log_info("AuditViewer initialized.")

    async def get_trace_by_event_id(
        self, event_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieves the full reasoning trace for a specific event ID.

        Args:
            event_id: The unique identifier for the event.

        Returns:
            The stored trace dictionary if found, otherwise None.
        """
        self.logger.log_debug(f"Audit: Retrieving trace for event_id: {event_id}")
        try:
            trace_data = await self.cot_database.retrieve_trace(event_id)
            if trace_data:
                self.logger.log_info(f"Audit: Found trace for event_id: {event_id}")
                return trace_data
            else:
                self.logger.log_warning(
                    f"Audit: No trace found for event_id: {event_id}"
                )
                return None
        except Exception as e:
            self.logger.log_error(
                f"Audit: Error retrieving trace for event_id {event_id}: {e}",
                exc_info=True,
            )
            return None

    async def query_traces_by_timestamp(
        self,
        start_time: datetime,
        end_time: datetime,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves all traces within a specific time window.

        Args:
            start_time: The start of the time window.
            end_time: The end of the time window.
            limit: The maximum number of traces to return.

        Returns:
            A list of trace dictionaries.
        """
        self.logger.log_debug(
            f"Audit: Querying traces from {start_time} to {end_time} (limit {limit})"
        )
        try:
            # This assumes cot_database has a method to query by timestamp.
            # If it's a simple key-value store, this might be inefficient
            # and require a different storage solution or indexing.
            # For this example, we'll assume a query method exists.
            traces = await self.cot_database.query_by_time(start_time, end_time, limit)
            self.logger.log_info(
                f"Audit: Found {len(traces)} traces for time window."
            )
            return traces
        except Exception as e:
            self.logger.log_error(
                f"Audit: Error querying traces by timestamp: {e}", exc_info=True
            )
            return []

    async def search_traces(
        self, keyword: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Searches traces for a specific keyword.
        Note: This is likely to be very inefficient on a simple DB.
        A real implementation would use a search-indexed database.

        Args:
            keyword: The keyword to search for (case-insensitive).
            limit: The maximum number of traces to return.

        Returns:
            A list of matching trace dictionaries.
        """
        self.logger.log_debug(
            f"Audit: Searching traces for keyword '{keyword}' (limit {limit})"
        )
        try:
            # This is a highly inefficient implementation for demonstration.
            # DO NOT use in production without a proper search index.
            all_keys = await self.cot_database.get_all_keys()
            matches = []
            keyword_lower = keyword.lower()

            for key in all_keys:
                if len(matches) >= limit:
                    break
                trace_data = await self.cot_database.retrieve_trace(key)
                if trace_data:
                    trace_str = json.dumps(trace_data).lower()
                    if keyword_lower in trace_str:
                        matches.append(trace_data)

            self.logger.log_info(
                f"Audit: Found {len(matches)} traces matching keyword '{keyword}'."
            )
            return matches
        except Exception as e:
            self.logger.log_error(
                f"Audit: Error searching traces for keyword '{keyword}': {e}",
                exc_info=True,
            )
            return []

    # Placeholder for query_logs in interfaces/api_server.py
    def query_logs(self, limit: int, component_filter: Optional[str]) -> List[Dict[str, Any]]:
        """
        Synchronous placeholder for the API server.
        In a real app, this might run an async query and block,
        or the API server would be async.
        """
        self.logger.log_warning("AuditViewer.query_logs is a synchronous placeholder.")
        # This is a mock implementation.
        # A real implementation would need to run the async query_traces
        # or have a different backend.
        return [
            {
                "timestamp": datetime.now().isoformat(),
                "component": component_filter or "mock_component",
                "message": f"Mock audit log {i}",
                "event_id": f"mock_event_{i}",
            }
            for i in range(min(limit, 5))
        ]

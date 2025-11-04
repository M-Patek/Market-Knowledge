from typing import Dict, Any, Optional, List
from datetime import datetime
import asyncio
import aiofiles
import json
import os

# 修正：将 'core.schemas...' 转换为 'Phoenix_project.core.schemas...'
from Phoenix_project.core.schemas.fusion_result import FusionResult
# 修正：将 'monitor.logging...' 转换为 'Phoenix_project.monitor.logging...'
from Phoenix_project.monitor.logging import ESLogger


class CoTDatabase:
    """
    A simple filesystem-based database to store the Chain-of-Thought (CoT)
    reasoning traces (FusionResult objects) for auditing and analysis.

    In a production system, this would be replaced by a robust database
    (e.g., MongoDB, BigQuery, or a dedicated audit store).
    """

    def __init__(self, config: Dict[str, Any], logger: ESLogger):
        """
        Initializes the CoTDatabase.

        Args:
            config: A dictionary containing configuration,
                    specifically `db_path`.
            logger: An instance of ESLogger for logging.
        """
        self.db_path = config.get("db_path", "./audit_traces")
        self.logger = logger
        self.lock = asyncio.Lock()

        # Ensure the database directory exists
        try:
            os.makedirs(self.db_path, exist_ok=True)
            self.logger.log_info(
                f"CoTDatabase initialized. Storage path: {self.db_path}"
            )
        except OSError as e:
            self.logger.log_error(
                f"Failed to create CoTDatabase directory at {self.db_path}: {e}",
                exc_info=True,
            )
            raise

    def _get_filepath(self, event_id: str) -> str:
        """Helper to get the file path for a given event ID."""
        # Sanitize event_id to prevent path traversal issues
        filename = "".join(c for c in event_id if c.isalnum() or c in ("-", "_", "."))
        if not filename:
            filename = f"invalid_event_id_{hash(event_id)}"
        return os.path.join(self.db_path, f"{filename}.json")

    async def store_trace(
        self, event_id: str, trace_data: Dict[str, Any]
    ) -> bool:
        """
        Stores the full reasoning trace (FusionResult) for a given event.

        Args:
            event_id: The unique identifier for the event.
            trace_data: The FusionResult object (as a dictionary).

        Returns:
            True if storage was successful, False otherwise.
        """
        if not event_id:
            self.logger.log_warning("store_trace called with empty event_id. Skipping.")
            return False

        filepath = self._get_filepath(event_id)
        self.logger.log_debug(f"Storing trace for event {event_id} to {filepath}")

        try:
            async with self.lock:
                async with aiofiles.open(filepath, "w", encoding="utf-8") as f:
                    # We serialize the Pydantic model (passed as dict)
                    await f.write(json.dumps(trace_data, indent=2, default=str))
            self.logger.log_info(f"Successfully stored trace for event {event_id}.")
            return True
        except Exception as e:
            self.logger.log_error(
                f"Failed to store trace for event {event_id}: {e}", exc_info=True
            )
            return False

    async def retrieve_trace(self, event_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the reasoning trace for a specific event ID.

        Args:
            event_id: The unique identifier for the event.

        Returns:
            The stored trace dictionary if found, otherwise None.
        """
        filepath = self._get_filepath(event_id)
        self.logger.log_debug(f"Retrieving trace for event {event_id} from {filepath}")

        if not os.path.exists(filepath):
            self.logger.log_warning(f"No trace found for event {event_id} at {filepath}")
            return None

        try:
            async with aiofiles.open(filepath, "r", encoding="utf-8") as f:
                data = await f.read()
            return json.loads(data)
        except Exception as e:
            self.logger.log_error(
                f"Failed to retrieve or parse trace for event {event_id}: {e}",
                exc_info=True,
            )
            return None

    # --- Methods for AuditViewer (Inefficient examples) ---

    async def query_by_time(
        self, start_time: datetime, end_time: datetime, limit: int
    ) -> List[Dict[str, Any]]:
        """
        Inefficiently queries traces by time.
        WARNING: This is a placeholder. It scales poorly.
        """
        self.logger.log_warning("query_by_time is highly inefficient on filesystem DB.")
        traces = []
        all_files = await self.get_all_keys()
        
        for event_id in all_files:
            if len(traces) >= limit:
                break
            trace = await self.retrieve_trace(event_id)
            if trace and "timestamp" in trace:
                try:
                    # Assuming timestamp is in a standard ISO format
                    ts_str = trace["timestamp"]
                    # Handle potential timezone info
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    
                    # Make sure times are comparable (e.g., all aware or all naive)
                    # This is a complex topic, simplified here.
                    if (
                        ts.tzinfo is None
                        and start_time.tzinfo is not None
                        and end_time.tzinfo is not None
                    ):
                        # A-hoc: assume trace timestamp is UTC if others are
                        ts = ts.replace(tzinfo=datetime.timezone.utc)
                    
                    if start_time <= ts <= end_time:
                        traces.append(trace)
                except Exception as e:
                    self.logger.log_warning(
                        f"Could not parse timestamp for event {event_id}: {e}"
                    )
        return traces

    async def get_all_keys(self) -> List[str]:
        """
        Inefficiently gets all event IDs (filenames).
        WARNING: This is a placeholder.
        """
        try:
            async with self.lock:
                files = [
                    f.replace(".json", "")
                    for f in os.listdir(self.db_path)
                    if f.endswith(".json")
                ]
            return files
        except Exception as e:
            self.logger.log_error(f"Failed to list all keys in CoTDatabase: {e}")
            return []

# Phoenix_project/snapshot_manager.py
# [Task P1-005] Snapshot Version Sync & Explicit Pydantic Fields.

import logging
from datetime import datetime
from omegaconf import DictConfig
from typing import Optional

from Phoenix_project.storage.s3_client import S3Client
from Phoenix_project.data_manager import DataManager 
from Phoenix_project.context_bus import ContextBus
from Phoenix_project.core.pipeline_state import PipelineState
from collections import deque 

# Restore Schemas
from Phoenix_project.core.schemas.data_schema import MarketData, NewsData, EconomicIndicator, PortfolioState
from Phoenix_project.core.schemas.fusion_result import AgentDecision, FusionResult

logger = logging.getLogger(__name__)

class SnapshotManager:
    """
    [已实现]
    负责创建和恢复系统状态的快照。
    用于在系统重启或回测时恢复状态。
    """

    def __init__(self, config: DictConfig, s3_client: S3Client, data_manager: DataManager, context_bus: ContextBus):
        self.config = config.get("snapshot_manager", {})
        self.s3_client = s3_client
        self.data_manager = data_manager
        self.context_bus = context_bus 
        
        self.snapshot_prefix = self.config.get("s3_prefix", "snapshots/")
        logger.info("SnapshotManager initialized.")

    def _get_state_sources(self) -> dict:
        """
        Gather state sources for snapshot.
        """
        logger.debug("Gathering state sources for snapshot...")
        state = {}
        
        try:
            # 1. Portfolio State
            if hasattr(self.data_manager, "get_current_portfolio"):
                state["portfolio"] = self.data_manager.get_current_portfolio()
            else:
                state["portfolio"] = None
            
            # 2. Market Data
            if hasattr(self.data_manager, "get_latest_market_data"):
                state["market_data"] = self.data_manager.get_latest_market_data()
            else:
                state["market_data"] = None

            # 3. ContextBus / PipelineState
            if hasattr(self, "context_bus") and self.context_bus:
                logger.debug("Gathering ContextBus state...")
                cb_state: PipelineState = self.context_bus.get_current_state()
                
                # Use Pydantic dump
                # Note: 'history' fields are now explicit in PipelineState, so model_dump handles them.
                # However, they might be huge, so we might want to trim them here if not for full restore.
                # For now, full dump.
                snap_dict = cb_state.model_dump(mode='json')
                
                # Add version explicitly
                snap_dict["__version__"] = cb_state.VERSION
                
                state["context_bus_snapshot"] = snap_dict
            else:
                logger.warning("ContextBus not injected. Skipping ContextBus snapshot.")
                state["context_bus_snapshot"] = None

            state["l3_drl_state"] = None # Skipped as per policy
            
        except Exception as e:
            logger.error(f"Failed to gather state sources: {e}", exc_info=True)
            return {}
            
        return state

    def create_snapshot(self, snapshot_name: str | None = None) -> str | None:
        """
        Create snapshot and save to S3.
        """
        state_data = self._get_state_sources()
        if not state_data:
            return None
            
        if snapshot_name is None:
            snapshot_name = f"snapshot_{datetime.now().isoformat().replace(':', '-')}.json"
        
        key = f"{self.snapshot_prefix}{snapshot_name}"
        logger.info(f"Creating snapshot at S3 key: {key}")
        
        success = self.s3_client.upload_json(key, state_data)
        
        if success:
            logger.info(f"Snapshot created successfully: {key}")
            return key
        else:
            logger.error(f"Failed to upload snapshot to S3: {key}")
            return None

    def load_snapshot(self, snapshot_name: str) -> bool:
        """
        Load snapshot from S3 and restore state.
        [Task P1-005] Version Check & Migration Hook.
        """
        key = f"{self.snapshot_prefix}{snapshot_name}"
        logger.info(f"Loading snapshot from S3 key: {key}")
        
        state_data = self.s3_client.load_json(key)
        
        if state_data is None:
            logger.error(f"Failed to load snapshot: Data not found or invalid JSON at {key}")
            return False

        try:
            # Restore DataManager components
            if "portfolio" in state_data and hasattr(self.data_manager, "load_portfolio_state"):
                self.data_manager.load_portfolio_state(state_data["portfolio"])
            
            if "market_data" in state_data and hasattr(self.data_manager, "load_market_data_state"):
                self.data_manager.load_market_data_state(state_data["market_data"])
            
            # Restore ContextBus State
            if "context_bus_snapshot" in state_data and self.context_bus:
                snap_data = state_data["context_bus_snapshot"]
                
                # [Task P1-005] Version Check
                snap_version = snap_data.get("__version__", 0)
                current_version = PipelineState.VERSION
                
                if snap_version != current_version:
                    logger.warning(f"Snapshot version mismatch! Snap: {snap_version}, Curr: {current_version}. Running migration...")
                    snap_data = self._migrate_state(snap_data, snap_version, current_version)
                
                # Remove version metadata before Pydantic load if not in schema
                snap_data.pop("__version__", None)
                
                # Reconstruct PipelineState
                try:
                    restored_state = PipelineState(**snap_data)
                    self.context_bus.update_state(restored_state)
                    logger.info("ContextBus state restored successfully.")
                except Exception as e:
                    logger.error(f"Pydantic validation failed during restore: {e}")
                    raise

            logger.info(f"Snapshot '{snapshot_name}' loaded and restored successfully.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore state from snapshot '{snapshot_name}': {e}", exc_info=True)
            return False

    def _migrate_state(self, data: dict, from_ver: int, to_ver: int) -> dict:
        """
        [Task P1-005] Simple migration logic.
        """
        # Example: if migrating from 0 to 1, ensure new fields exist
        if from_ver < 1:
            if "market_data_history" not in data: data["market_data_history"] = []
            if "news_history" not in data: data["news_history"] = []
            if "econ_history" not in data: data["econ_history"] = []
            if "decision_history" not in data: data["decision_history"] = []
            if "fusion_history" not in data: data["fusion_history"] = []
        
        return data

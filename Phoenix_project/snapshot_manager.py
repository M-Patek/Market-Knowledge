# snapshot_manager.py
"""
Manages the creation and verification of immutable data snapshots for
ensuring 100% reproducibility of backtests and experiments.
"""
import os
import shutil
import json
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

class SnapshotManager:
    """
    Handles the creation, management, and verification of data snapshots.
    """
    def __init__(self, base_dir: str = "snapshots", cache_dir: str = "data_cache"):
        self.base_dir = Path(base_dir)
        self.cache_dir = Path(cache_dir)
        self.logger = logging.getLogger("PhoenixProject.SnapshotManager")
        self.logger.info(f"SnapshotManager initialized. Base directory: '{self.base_dir}'")

    def _calculate_file_hash(self, filepath: Path) -> str:
        """Calculates the SHA256 hash of a file's content."""
        h = hashlib.sha256()
        with open(filepath, 'rb') as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()

    def _create_snapshot_directory(self, snapshot_path: Path):
        """Creates the directory for the new snapshot."""
        snapshot_path.mkdir(parents=True, exist_ok=False)
        self.logger.info(f"Created new snapshot directory: '{snapshot_path}'")

    def _copy_files_and_build_manifest_list(self, required_files: List[str], snapshot_path: Path) -> List[Dict[str, str]]:
        """Copies required files to the snapshot and returns a list for the manifest."""
        manifest_files = []
        for filename in required_files:
            source_path = self.cache_dir / filename
            dest_path = snapshot_path / filename

            if source_path.exists():
                shutil.copy2(source_path, dest_path)
                file_hash = self._calculate_file_hash(dest_path)
                manifest_files.append({
                    "filename": filename,
                    "sha256_hash": file_hash
                })
            else:
                self.logger.warning(f"Required data file '{filename}' not found in cache. Skipping.")
        return manifest_files

    def _write_manifest(self, manifest_path: Path, manifest_data: Dict[str, Any]):
        """Writes the manifest JSON file."""
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest_data, f, indent=2)

    def create_snapshot(self, run_id: str, required_files: List[str]) -> str:
        """
        Creates a new, immutable snapshot of all data required for a specific run.

        Args:
            run_id: The unique identifier for the current experimental run.
            required_files: A list of filenames from the data_cache that are needed.

        Returns:
            The ID of the created snapshot.
        """
        snapshot_id = f"snapshot_{run_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        snapshot_path = self.base_dir / snapshot_id
        try:
            self._create_snapshot_directory(snapshot_path)

            manifest = {
                "snapshot_id": snapshot_id,
                "created_utc": datetime.utcnow().isoformat(),
                "run_id": run_id,
                "files": self._copy_files_and_build_manifest_list(required_files, snapshot_path)
            }

            self._write_manifest(snapshot_path / "manifest.json", manifest)

            self.logger.info(f"Successfully created data snapshot with ID: '{snapshot_id}'")
            return snapshot_id

        except Exception as e:
            self.logger.error(f"Failed to create data snapshot '{snapshot_id}': {e}")
            # Clean up partially created snapshot
            if snapshot_path.exists():
                shutil.rmtree(snapshot_path)
            raise

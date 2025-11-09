# Phoenix_project/snapshot_manager.py
import os
import shutil
import logging
import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class SnapshotManager:
    """
    Manages creating and restoring system state snapshots.
    (TBD: 这是用于什么状态? 内存? 投资组合? 完整系统?)
    (假设: 它用于 L1/L2 智能体的内存/状态, 如 RAG DBs, CoT)
    """
    def __init__(self, config):
        self.config = config
        self.snapshot_dir = Path(config.paths.snapshots)
        self.cache_dir = Path(config.paths.cache) # (TBD: 这应该由快照管理吗?)
        
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        # (我们不在 init 中创建 cache_dir, restore 应该创建它)

    def create_snapshot(self) -> str:
        """
        Creates a snapshot of the current system state (e.g., memory).
        Returns the path to the created snapshot file.
        """
        
        # (TBD: 'state_sources' 应该由配置驱动)
        # (假设我们正在快照 cache_dir)
        state_source_dir = self.cache_dir
        
        if not state_source_dir.exists():
            logger.warning(f"State source directory not found: {state_source_dir}. Cannot create snapshot.")
            return None

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_name = f"snapshot_{timestamp}"
        snapshot_archive_path = self.snapshot_dir / snapshot_name
        
        try:
            logger.info(f"Creating snapshot of {state_source_dir}...")
            
            # (我们创建 .zip 格式)
            archive_path_str = shutil.make_archive(
                base_name=str(snapshot_archive_path),
                format="zip",
                root_dir=state_source_dir.parent,
                base_dir=state_source_dir.name
            )
            
            logger.info(f"Snapshot created successfully: {archive_path_str}")
            return archive_path_str
            
        except Exception as e:
            logger.error(f"Failed to create snapshot: {e}", exc_info=True)
            return None

    def restore_snapshot(self, snapshot_path: str) -> bool:
        """
        Restores the system state from a given snapshot file.
        (这会清除当前的 cache_dir 并用快照内容替换它)
        """
        snapshot_file = Path(snapshot_path)
        if not snapshot_file.exists():
            logger.error(f"Snapshot file not found: {snapshot_path}")
            return False
            
        # 1. 清理当前的 cache_dir
        if self.cache_dir.exists():
            logger.warning(f"Removing existing cache directory: {self.cache_dir}")
            try:
                shutil.rmtree(self.cache_dir)
            except Exception as e:
                logger.error(f"Failed to remove existing cache: {e}. Aborting restore.", exc_info=True)
                return False
        
        # 2. 从归档中恢复
        try:
            logger.info(f"Restoring snapshot from {snapshot_path} to {self.cache_dir}...")
            
            # [主人喵的修复 11.10] 取消模拟，实际解压快照
            # (我们指定 cache_dir 的 *父级* 作为解压目标,
            # 因为归档文件包含 'cache' 目录本身)
            # (不, make_archive 的 base_dir=state_source_dir.name 意味着
            # .zip 的根目录是 'cache'。所以我们解压到 cache_dir 的 *父级*)
            
            # (让我们重新考虑一下 make_archive 的逻辑...)
            # root_dir=state_source_dir.parent, base_dir=state_source_dir.name
            # 这将创建一个 .zip，其中包含一个名为 'cache' (或 cache_dir.name) 的顶级目录。
            
            # 因此, unpack_archive 应该解压到 state_source_dir.parent
            extraction_target_dir = self.cache_dir.parent
            
            shutil.unpack_archive(snapshot_path, extraction_target_dir)
            
            # [主人喵的修复 11.10] 移除了模拟代码
            # Mocking the outcome
            # os.makedirs(self.cache_dir, exist_ok=True) 
            
            # (TBD: 验证快照完整性)
            if not self.cache_dir.exists():
                raise RuntimeError(f"Extraction completed, but target directory {self.cache_dir} was not created.")

            logger.info(f"Snapshot restored successfully to {self.cache_dir}.")
            return True
            
        except FileNotFoundError:
            # (这不应该发生, 因为我们已经检查过了, 但以防万一)
            logger.error(f"Snapshot file not found during restore: {snapshot_path}")
            return False
        except Exception as e:
            logger.error(f"Failed to restore snapshot: {e}", exc_info=True)
            # 尝试清理部分解压
            if self.cache_dir.exists():
                logger.warning("Cleaning up partially restored cache directory...")
                try:
                    shutil.rmtree(self.cache_dir)
                except Exception as cleanup_e:
                    logger.error(f"Failed to cleanup partial restore: {cleanup_e}")
            return False

    def list_snapshots(self) -> list[str]:
        """Lists available snapshots."""
        try:
            snapshots = [f.name for f in self.snapshot_dir.glob("snapshot_*.zip")]
            snapshots.sort(reverse=True) # (最新的在最前面)
            return snapshots
        except Exception as e:
            logger.error(f"Failed to list snapshots: {e}", exc_info=True)
            return []

    def get_latest_snapshot(self) -> str | None:
        """Gets the path to the most recent snapshot."""
        snapshots = self.list_snapshots()
        if snapshots:
            return str(self.snapshot_dir / snapshots[0])
        return None

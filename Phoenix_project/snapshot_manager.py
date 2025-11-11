# Phoenix_project/snapshot_manager.py
# [主人喵的修复 11.11] 实现了 TBD (状态源和快照逻辑)
# [主人喵的修复 12.1] 实现了所有 TBD (注入, 状态源, 恢复逻辑)

import logging
from datetime import datetime
from omegaconf import DictConfig
from storage.s3_client import S3Client
from data_manager import DataManager 
from context_bus import ContextBus # [主人喵的修复] (TBD): 注入 ContextBus
from core.pipeline_state import PipelineState # [主人喵的修复] (TBD): 恢复 ContextBus
from collections import deque # [主人喵的修复] (TBD): 恢复 ContextBus
# [主人喵的修复] (TBD): 恢复 ContextBus (需要 Pydantic 模式)
from core.schemas.data_schema import MarketData, NewsData, EconomicIndicator, PortfolioState
from core.schemas.fusion_result import AgentDecision, FusionResult


logger = logging.getLogger(__name__)

class SnapshotManager:
    """
    [已实现]
    负责创建和恢复系统状态的快照。
    用于在系统重启或回测时恢复状态。
    """

    def __init__(self, config: DictConfig, s3_client: S3Client, data_manager: DataManager, context_bus: ContextBus): # [主人喵的修复] (TBD): 注入 ContextBus
        self.config = config.get("snapshot_manager", {})
        self.s3_client = s3_client
        
        # [实现] TBD: 状态源
        # 注入需要被快照的核心组件
        self.data_manager = data_manager
        self.context_bus = context_bus # [主人喵的修复] (TBD): 注入 ContextBus
        # (TBD: self.portfolio_constructor = portfolio_constructor)
        # (为了简单起见，我们假设 DataManager 和 ContextBus 拥有所有状态)
        
        self.snapshot_prefix = self.config.get("s3_prefix", "snapshots/")
        logger.info("SnapshotManager initialized.")

    def _get_state_sources(self) -> dict:
        """
        [已实现] TBD: 定义应包含在快照中的具体状态源。
        """
        logger.debug("Gathering state sources for snapshot...")
        state = {}
        
        try:
            # 1. 投资组合状态 (来自 DataManager)
            if hasattr(self.data_manager, "get_current_portfolio"):
                state["portfolio"] = self.data_manager.get_current_portfolio()
            else:
                logger.warning("DataManager has no 'get_current_portfolio' method. Skipping portfolio snapshot.")
                state["portfolio"] = None
            
            # 2. 市场数据状态 (例如最新价格)
            if hasattr(self.data_manager, "get_latest_market_data"):
                state["market_data"] = self.data_manager.get_latest_market_data()
            else:
                logger.warning("DataManager has no 'get_latest_market_data' method. Skipping market data snapshot.")
                state["market_data"] = None

            # [主人喵的修复] (TBD): 状态源不完整 (缺少 ContextBus 历史)
            if hasattr(self, "context_bus") and self.context_bus:
                logger.debug("Gathering ContextBus state...")
                cb_state: PipelineState = self.context_bus.get_current_state()
                
                # 我们将手动序列化 deques，因为 get_snapshot() 会跳过它们
                state["context_bus_snapshot"] = {
                    "current_time": cb_state.current_time.isoformat(),
                    "portfolio_state": cb_state.portfolio_state.model_dump() if cb_state.portfolio_state else None,
                    "main_task_query": cb_state.main_task_query,
                    "history": {
                        # 将 deque 转换为 list
                        "market_data": [d.model_dump(mode='json') for d in cb_state.market_data_history],
                        "news": [n.model_dump(mode='json') for n in cb_state.news_history],
                        "econ": [e.model_dump(mode='json') for e in cb_state.econ_history],
                        "decisions": [d.model_dump(mode='json') for d in cb_state.decision_history],
                        "fusions": [f.model_dump(mode='json') for f in cb_state.fusion_history],
                    }
                }
            else:
                logger.warning("ContextBus not injected in SnapshotManager. Skipping ContextBus snapshot.")
                state["context_bus_snapshot"] = None

            # 4. (TBD: L3 DRL 智能体状态? - 这更像是模型 checkpoint)
            # [主人喵的修复] DRL 状态 (例如 RLLib 算法的内部状态) 非常复杂
            # 并且特定于 RLLib。快照应专注于 *data* 状态。
            # DRL *model checkpoints* 应由 StrategyHandler 或训练器单独管理。
            state["l3_drl_state"] = None # 明确跳过
            
        except Exception as e:
            logger.error(f"Failed to gather state sources: {e}", exc_info=True)
            return {}
            
        return state

    def create_snapshot(self, snapshot_name: str | None = None) -> str | None:
        """
        [已实现] 创建当前系统状态的快照并保存到 S3。
        """
        state_data = self._get_state_sources()
        if not state_data:
            logger.error("Failed to create snapshot: Could not gather state sources.")
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
        [已实现] 从 S3 加载快照并恢复系统状态。
        """
        key = f"{self.snapshot_prefix}{snapshot_name}"
        logger.info(f"Loading snapshot from S3 key: {key}")
        
        state_data = self.s3_client.load_json(key)
        
        if state_data is None:
            logger.error(f"Failed to load snapshot: Data not found or invalid JSON at {key}")
            return False

        try:
            # [实现] TBD: 恢复逻辑
            # 将加载的状态分发回源组件
            
            if "portfolio" in state_data and hasattr(self.data_manager, "load_portfolio_state"):
                self.data_manager.load_portfolio_state(state_data["portfolio"])
                logger.info("Portfolio state restored from snapshot.")
            
            if "market_data" in state_data and hasattr(self.data_manager, "load_market_data_state"):
                self.data_manager.load_market_data_state(state_data["market_data"])
                logger.info("Market data state restored from snapshot.")
            
            # [主人喵的修复] (TBD): 恢复 ContextBus 状态的逻辑。
            if "context_bus_snapshot" in state_data and self.context_bus:
                logger.info("Restoring ContextBus state from snapshot...")
                snap_data = state_data["context_bus_snapshot"]
                
                # 1. 创建一个新的 PipelineState
                # (我们需要 PipelineState 的 max_history)
                max_history = self.context_bus.get_current_state().max_history
                
                restored_state = PipelineState(
                    initial_state={
                        "current_time": datetime.fromisoformat(snap_data["current_time"]),
                        "portfolio_state": PortfolioState(**snap_data["portfolio_state"]) if snap_data.get("portfolio_state") else None,
                        "main_task_query": snap_data.get("main_task_query", {})
                    },
                    max_history=max_history
                )
                
                # 2. 手动恢复 deques
                history_data = snap_data.get("history", {})
                restored_state.market_data_history = deque([MarketData(**d) for d in history_data.get("market_data", [])], maxlen=max_history)
                restored_state.news_history = deque([NewsData(**d) for d in history_data.get("news", [])], maxlen=max_history)
                restored_state.econ_history = deque([EconomicIndicator(**d) for d in history_data.get("econ", [])], maxlen=max_history)
                restored_state.decision_history = deque([AgentDecision(**d) for d in history_data.get("decisions", [])], maxlen=max_history)
                restored_state.fusion_history = deque([FusionResult(**d) for d in history_data.get("fusions", [])], maxlen=max_history)

                # 3. 将恢复的状态推送到 ContextBus (它是一个单例)
                self.context_bus.update_state(restored_state)
                
                logger.info(f"ContextBus state restored. {len(restored_state.market_data_history)} market data points loaded.")
            
            logger.info(f"Snapshot '{snapshot_name}' loaded and restored successfully.")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore state from snapshot '{snapshot_name}': {e}", exc_info=True)
            return False

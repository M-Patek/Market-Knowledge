# Phoenix_project/phoenix_project.py
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os

# [主人喵的修复] 导入日志配置模块
import logging.config
import hydra.utils

# [主人喵的修复] (TBD 已解决):
# (DRL, run_training, TradingEnv.reset 的 TBD 注释已移至其各自的文件
# 或已在此次更新中解决)


# [主人喵的清洁计划 5.2] 导入注册表
from registry import Registry
from controller.loop_manager import LoopManager
from snapshot_manager import SnapshotManager
from storage.s3_client import S3Client
from data_manager import DataManager
from context_bus import ContextBus

# [主人喵的清洁计划 5.2]
# [主人喵的修复] (TBD 已解决): Hydra 日志配置。
# 我们依赖 Hydra 通过 config/system.yaml 来配置日志记录。
# 我们在这里获取一个记录器，Hydra 将自动对其进行配置。
logger = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name="system", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Phoenix (Market Knowledge)
    
    (这是对 5.2 计划的重构)
    """
    
    # [主人喵的修复] (TBD 已解决): Hydra 日志配置。
    # Hydra 会自动根据 'logging' 部分配置 config/system.yaml。
    # 我们也可以在这里使用 hydra.utils.log 来获取已配置的记录器。
    # log = hydra.utils.log
    # log.info("Hydra logging configured.")

    logger.info("Initializing Phoenix System...")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.debug(f"Full configuration:\n{OmegaConf.to_yaml(cfg)}")

    # 1. 初始化核心组件注册表
    # (Registry 现在负责构建核心服务)
    try:
        registry = Registry(cfg)
    except Exception as e:
        logger.exception(f"Fatal error during Registry initialization: {e}")
        return # 无法继续

    # 2. 构建完整的系统管道
    # (这会实例化所有智能体和引擎)
    try:
        system = registry.build_system(cfg)
    except Exception as e:
        logger.exception(f"Fatal error during system build (build_system): {e}")
        return # 无法继续

    # 3. 初始化 S3 和快照 (如果需要)
    # (注意：S3Client 现在可能应该在 Registry 中初始化)
    # (为了保持 5.2 的结构，我们暂时在这里初始化)
    s3_client = S3Client(cfg.s3)
    
    # [主人喵的修复 12.1] SnapshotManager 现在需要 ContextBus 和 DataManager
    snapshot_manager = SnapshotManager(
        config=cfg,
        s3_client=s3_client,
        data_manager=system.data_manager, # 从构建的系统中获取
        context_bus=system.context_bus      # 从构建的系统中获取
    )

    # 4. (TBD) 加载快照 (如果配置了)
    if cfg.main.load_snapshot:
        logger.info(f"Attempting to load snapshot: {cfg.main.snapshot_name}")
        success = snapshot_manager.load_snapshot(cfg.main.snapshot_name)
        if not success:
            logger.error(f"Failed to load snapshot {cfg.main.snapshot_name}. Starting with fresh state.")
        else:
            logger.info(f"Snapshot {cfg.main.snapshot_name} loaded successfully.")
    else:
        logger.info("Starting with a fresh state (no snapshot loaded).")


    # 5. 初始化主循环管理器
    # (LoopManager 现在从注册表中获取构建好的组件)
    loop_manager = LoopManager(
        config=cfg.main_loop,
        orchestrator=system.orchestrator,
        data_iterator=system.data_iterator, # [主人喵的修复] 注入 DataIterator
        context_bus=system.context_bus,
        snapshot_manager=snapshot_manager
    )

    # 6. 运行系统
    try:
        logger.info("Starting Phoenix main loop...")
        if cfg.main.mode == "backtest":
            loop_manager.run_backtest_loop()
        elif cfg.main.mode == "live":
            # (TBD: 实现实时循环) - [FIX Applied]
            loop_manager.run_live_loop()
        else:
            logger.error(f"Unknown main mode: {cfg.main.mode}")
            
    except KeyboardInterrupt:
        logger.info("Phoenix system interrupted by user (Ctrl+C). Shutting down...")
    except Exception as e:
        logger.exception(f"Unhandled exception in main loop: {e}")
    finally:
        logger.info("Phoenix system shutting down...")
        # (TBD: 清理资源，例如数据库连接) - [FIX Applied]
        try:
            if 'registry' in locals() and registry:
                registry.cleanup_resources()
            logger.info("Phoenix system shutdown complete.")
        except Exception as e:
            logger.exception(f"Error during resource cleanup: {e}")
            logger.info("Phoenix system shutdown complete despite cleanup error.")


if __name__ == "__main__":
    main()

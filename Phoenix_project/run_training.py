"""
[阶段 4] 已修改
DRL (深度强化学习) 训练执行脚本。

现在包含 ET 感知的安全停止逻辑，与 gnn_engine.py 类似。
[Phase 5] Added Backtest Mode and Dry Run.
"""

import argparse
import os
import datetime # [阶段 4]
import pytz # [阶段 4]
import time # [阶段 4]
import asyncio
from unittest.mock import MagicMock

from Phoenix_project.training.drl.multi_agent_trainer import MultiAgentDRLTrainer
from Phoenix_project.training.walk_forward_trainer import WalkForwardTrainer
from Phoenix_project.config.loader import ConfigLoader # [Fix II.2]
from Phoenix_project.monitor.logging import get_logger
# [阶段 4] 导入安全逻辑依赖
from Phoenix_project.monitor.metrics import METRICS
from Phoenix_project.models.registry import registry, MODEL_ARTIFACTS_DIR
# [Phase 5] Import for Backtest Mode
from Phoenix_project.training.backtest_engine import BacktestEngine
from Phoenix_project.core.pipeline_state import PipelineState
from Phoenix_project.core.schemas.data_schema import MarketData

logger = get_logger(__name__)

# [阶段 4] 关键时区和安全停止时间
MARKET_TZ = pytz.timezone("America/New_York")
# 09:00 ET (在 09:30 ET 开市前 30 分钟)
SAFE_STOP_TIME_ET = datetime.time(9, 0, 0)

def _check_et_safety_window() -> bool:
    """
    [阶段 4] 检查是否已进入 ET 危险窗口 (市场即将开市)。
    (与 gnn_engine.py 中的逻辑相同)
    
    Returns:
        bool: True 表示已进入危险窗口 (应停止), False 表示安全 (可继续)。
    """
    now_et = datetime.datetime.now(MARKET_TZ)
    is_weekday = 0 <= now_et.weekday() <= 4
    is_danger_window = is_weekday and (now_et.time() >= SAFE_STOP_TIME_ET)
    return is_danger_window

def run_fine_tuning(config: ConfigLoader): # [Fix II.2]
    """
    [阶段 4] 已修改
    执行 DRL 模型的夜间微调 (fine-tuning)。
    由 Celery (worker.py) 在 GNN 训练 *之后* 调用。
    """
    logger.info("[DRL Training] Nightly DRL fine-tuning STARTED.")
    
    # [自动回退] 
    # 1. 加载*当前生产*模型 (前一天的) 作为微调的起点
    # (注意: registry.load_production_model 返回的是模型对象,
    # DRLTrainer 可能需要一个路径或特定的加载方式。
    # 为简单起见，我们假设 DRLTrainer 可以接受 'None' 并自行加载)
    
    # 现实中:
    # base_model_path = registry.get_production_model_path("drl")
    # if not base_model_path:
    #     logger.warning("No base DRL model found, starting from scratch (if configured).")
    #     base_model_path = None # DRLTrainer 会处理
    
    # 模拟 DRLTrainer
    # trainer = MultiAgentDRLTrainer(config=config, base_model_path=base_model_path)
    
    logger.info("[DRL Training] DRLTrainer initialized (loading previous model).")

    try:
        # 2. 模拟训练循环
        num_timesteps_total = 5_000_000
        num_steps_per_check = 100_000
        training_success = False

        for step in range(0, num_timesteps_total, num_steps_per_check):
            
            # [阶段 4] ET 感知安全逻辑
            if _check_et_safety_window():
                logger.critical(
                    "[DRL Training] ET-Aware Safety Stop: "
                    f"已进入 {SAFE_STOP_TIME_ET.strftime('%H:%M')} ET 危险窗口 (市场即将开市)。"
                    "安全停止 DRL 训练。"
                )
                METRICS.increment_counter("nightly_pipeline_timeout_total", tags={"task": "drl_training"})
                training_success = False
                break # 退出训练循环

            # 模拟训练
            logger.info(f"[DRL Training] Running DRL steps {step} to {step + num_steps_per_check}...")
            # (真实场景: trainer.learn(timesteps=num_steps_per_check))
            time.sleep(1) # 模拟工作
            
        else:
            # 循环正常完成
            logger.info("[DRL Training] DRL training completed successfully (not stopped).")
            training_success = True

        # 3. 暂存 (Save) 和 生效 (Promote)
        if training_success:
            logger.info("[DRL Training] Training successful. Proceeding to save and promote.")
            
            timestamp_str = datetime.datetime.now(pytz.UTC).strftime("%Y%m%dT%H%M")
            candidate_path = os.path.join(MODEL_ARTIFACTS_DIR, f"drl_candidate_{timestamp_str}.zip") # (SB3 通常用 .zip)

            # 1. 暂存 (Save)
            # (真实场景: trainer.save_model(candidate_path))
            logger.info(f"Saving mock DRL model to {candidate_path}")
            with open(candidate_path, 'w') as f:
                f.write(f"Mock DRL Model. Trained at: {timestamp_str}")

            # 2. 生效 (Promote) - [阶段 2]
            # 原子性更新。
            registry.promote_model("drl", candidate_path)
            
            logger.info(f"[DRL Training] Successfully promoted new DRL model: {candidate_path}")

        else:
            logger.warning("[DRL Training] DRL training was stopped (timeout) or failed. Model will NOT be promoted.")
            # [自动回退]：注册表 (Registry) 仍指向旧模型。

        logger.info("[DRL Training] Nightly DRL fine-tuning FINISHED.")
    
    except Exception as e:
        logger.error(f"[DRL Training] CRITICAL FAILURE in DRL pipeline: {e}", exc_info=True)
        # 向上抛出异常，以便 Celery 链 (chain) 知道它失败了
        raise

def run_backtest(config: ConfigLoader, dry_run: bool = False):
    """
    [Phase 5] Execute System Backtest / Dry Run.
    """
    logger.info(f"Starting Backtest (Dry Run: {dry_run})...")
    
    if dry_run:
        # Initialize BacktestEngine with Mocks for System Integrity Check
        mock_dm = MagicMock()
        mock_pipeline = PipelineState()
        mock_cognitive = MagicMock()
        mock_risk = MagicMock()
        
        engine = BacktestEngine(
            config={},
            data_manager=mock_dm,
            pipeline_state=mock_pipeline,
            cognitive_engine=mock_cognitive,
            risk_manager=mock_risk
        )
        
        # Mock Data Iterator
        async def mock_iterator():
            start_time = datetime.datetime.now(pytz.UTC)
            for i in range(5): # 5 Steps
                current_time = start_time + datetime.timedelta(minutes=i)
                # Mock Market Data Batch
                md = MarketData(
                    symbol="BTC/USD", 
                    timestamp=current_time, 
                    open=100.0, high=105.0, low=95.0, close=102.0, volume=1000.0
                )
                batch = {"market_data": [md]}
                yield current_time, batch
                
        # Run Loop
        try:
            asyncio.run(engine.run_backtest(mock_iterator()))
            logger.info("Dry Run Backtest completed successfully.")
        except Exception as e:
            logger.critical(f"Dry Run Failed: {e}", exc_info=True)
            raise
            
    else:
        # Full Backtest logic (not implemented in this phase)
        logger.warning("Full backtest mode not yet implemented. Use --dry-run.")


def run_walk_forward(config: ConfigLoader): # [Fix II.2]
    """执行步进优化 (Walk-Forward Optimization)"""
    logger.info("Starting Walk-Forward Optimization (WFO)...")
    wfo_trainer = WalkForwardTrainer(config)
    wfo_trainer.run_optimization_loop()
    logger.info("Walk-Forward Optimization FINISHED.")

def main():
    """主执行函数"""
    parser = argparse.ArgumentParser(description="Phoenix 训练执行器")
    parser.add_argument(
        "mode",
        choices=["fine-tune", "walk-forward", "full-retrain", "backtest"],
        help="要执行的训练模式 (fine-tune, walk-forward, full-retrain)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Run a quick system check.")
    args = parser.parse_args()

    # 加载主配置
    # [Fix II.2] 使用 ConfigLoader 完整构造函数
    config = ConfigLoader(
        system_config_path="config/system.yaml",
        rules_config_path="config/symbolic_rules.yaml"
    )
    
    if args.mode == "fine-tune":
        run_fine_tuning(config)
        
    elif args.mode == "walk-forward":
        run_walk_forward(config)
        
    elif args.mode == "full-retrain":
        logger.info("Full retraining (DRL) requested...")
        # (这可能是一个更长的过程, 暂时指向 fine-tune)
        run_fine_tuning(config)
        
    elif args.mode == "backtest":
        run_backtest(config, dry_run=args.dry_run)
    
    else:
        logger.error(f"未知的训练模式: {args.mode}")

if __name__ == "__main__":
    main()

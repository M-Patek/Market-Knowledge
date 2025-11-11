# Phoenix_project/run_training.py
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os

from registry import Registry
from training.drl.multi_agent_trainer import MultiAgentDRLTrainer
from training.walk_forward_trainer import WalkForwardTrainer
from training.backtest_engine import BacktestEngine
from data.data_iterator import DataIterator


# [主人喵的修复] (TBD 已解决): 
# (非 DRL 训练的数据加载和引擎初始化在下方实现)
# (DRL 训练的数据加载和环境初始化在下方实现)

logger = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name="system", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for training loops (DRL or Walk-Forward).
    """
    logger.info(f"Starting training run in mode: {cfg.training.mode}")
    logger.info(f"Current working directory: {os.getcwd()}")
    
    # 1. [主人喵的修复] (TBD 已解决) 初始化 Registry 和核心组件
    # (这对于 DRL (需要 DataIterator) 和非 DRL (需要 BacktestEngine) 都需要)
    registry = Registry(cfg)
    system = registry.build_system(cfg)
    
    # [主人喵的修复] (TBD 已解决) 获取 DataIterator
    data_iterator = system.data_iterator

    if cfg.training.mode == "drl":
        logger.info("Initializing DRL Multi-Agent Trainer...")
        # [主人喵的修复] (TBD 已解决): DRL 训练 (run_training) 的数据加载和环境初始化。
        
        # 1. (TBD) 创建 DRL (TradingEnv) 环境
        # (TradingEnv 将在 DRLTrainer 内部创建，但需要 data_iterator)
        
        # 2. [主人喵的修复] (TBD 已解决) 初始化 DRL 训练器
        drl_trainer = MultiAgentDRLTrainer(
            config=cfg.training.drl,
            data_iterator=data_iterator # (TBD 已解决) 注入 data_iterator
        )
        
        # 3. [主人喵的修复] (TBD 已解决) 运行 DRL 训练
        drl_trainer.train()
        
    elif cfg.training.mode == "walk_forward":
        logger.info("Initializing Walk-Forward Trainer...")
        
        # [主人喵的修复] (TBD 已解决): 非 DRL 训练 (run_training) 的数据加载和引擎初始化。
        
        # 1. [主人喵的修复] (TBD 已解决) 初始化 BacktestEngine
        # (它需要 portfolio_constructor, risk_manager 等)
        backtest_engine = BacktestEngine(
            cognitive_engine=system.cognitive_engine,
            data_iterator=data_iterator,
            context_bus=system.context_bus
        )
        
        # 2. [主人喵的修复] (TBD 已解决) 初始化 WalkForwardTrainer
        wf_trainer = WalkForwardTrainer(
            config=cfg.training.walk_forward,
            backtest_engine=backtest_engine
            # (TBD: 可能需要注入其他组件，如 Optimizer)
        )
        
        # 3. [主人喵的修复] (TBD 已解决) 运行 WalkForward 训练/优化
        wf_trainer.run_optimization()
        
    else:
        logger.error(f"Unknown training mode: {cfg.training.mode}")

if __name__ == "__main__":
    main()

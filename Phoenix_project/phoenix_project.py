# Phoenix_project/phoenix_project.py
# [主人喵的修复 11.11] 清理了 TBDs 并集成了 StrategyHandler。

import logging
import time
from omegaconf import DictConfig
import hydra

from registry import Registry
from data_manager import DataManager
from data.data_iterator import DataIterator
from training.engine import TrainingEngine

# (主人喵的清洁计划 5.3) [新]
from agents.l3.alpha_agent import AlphaAgent
from agents.l3.risk_agent import RiskAgent
from agents.l3.execution_agent import ExecutionAgent
from training.drl.trading_env import TradingEnv
from training.drl.multi_agent_trainer import MultiAgentDRLTrainer


logger = logging.getLogger(__name__)

class PhoenixProject:
    """
    Main application class for the Phoenix Project.
    """
    def __init__(self, config: DictConfig):
        self.config = config
        self.registry = Registry(config)
        self.system = None # Lazily initialized

    def setup(self):
        """Initializes all core components."""
        logger.info("Setting up Phoenix system...")
        self.system = self.registry.build_system(self.config)
        logger.info("System setup complete.")

    def run_backtest(self):
        """Runs the system in backtesting mode."""
        if not self.system:
            self.setup()
            
        logger.info("Starting Phoenix system in BACKTEST mode...")
        
        # 1. Get the data iterator from the registry (which gets it from DataManager)
        data_iterator = self.system.data_iterator
        if not data_iterator:
            logger.error("Failed to get DataIterator. Aborting backtest.")
            return

        # 2. Get the main orchestrator
        orchestrator = self.system.orchestrator

        # 3. Loop through data and run pipeline
        try:
            for timestamp, data_batch in data_iterator:
                logger.debug(f"Processing data for timestamp: {timestamp}")
                
                # The orchestrator runs the full L1-L2-L3 pipeline
                pipeline_state = orchestrator.run_pipeline(timestamp, data_batch)
                
                # [已澄清] 最终状态会在此处记录。
                # "存储" (Store) 功能由 MetricsCollector 或 AuditManager (如果已配置) 处理。
                if pipeline_state.final_decision:
                    logger.info(f"Final decision at {timestamp}: {pipeline_state.final_decision.action}")
                
                # (主人喵的清洁计划 5.3) [已澄清]
                # 假设: Orchestrator.run_pipeline(...) 内部
                # 已经调用了 L3、PortfolioConstructor 和 OrderManager。
                # L3 Agent (Execution) 会将信号发送到 OrderManager。
                # OrderManager 会更新 ContextBus 上的投资组合状态。
                # PipelineState 会反映这个最终结果。

        except Exception as e:
            logger.critical(f"Unhandled exception during backtest: {e}", exc_info=True)
        finally:
            logger.info("Backtest finished.")

    def run_drl_backtest(self):
        """
        (主人喵的清洁计划 5.3) [新]
        Runs the system in DRL backtesting mode.
        """
        if not self.system:
            self.setup()
        
        logger.info("Starting Phoenix system in DRL BACKTEST mode...")
        
        data_iterator = self.system.data_iterator
        if not data_iterator:
            logger.error("Failed to get DataIterator. Aborting DRL backtest.")
            return

        # [主人喵的修复 11.11] 使用 StrategyHandler 加载和激活 DRL 策略
        # (这取代了旧的 TBD 和硬编码的模型路径)
        strategy_name = self.config.get("drl_models", {}).get("active_strategy", "drl_v1")
        logger.info(f"Attempting to load DRL strategy: {strategy_name}")

        if not hasattr(self.system, "strategy_handler"):
             logger.error("StrategyHandler not found in system. Aborting.")
             return

        if not self.system.strategy_handler.load_strategy(strategy_name):
            logger.error(f"Failed to load strategy '{strategy_name}'. Aborting DRL backtest.")
            return
        if not self.system.strategy_handler.activate_strategy(strategy_name):
             logger.error(f"Failed to activate strategy '{strategy_name}'. Aborting DRL backtest.")
             return
        
        logger.info(f"Strategy '{strategy_name}' loaded and activated via StrategyHandler.")


        # 1. 加载 L3 DRL 智能体 (实例)
        # (模型权重已由 StrategyHandler 加载)
        l3_alpha_agent = self.system.l3_agents["alpha"]
        l3_risk_agent = self.system.l3_agents["risk"]
        l3_execution_agent = self.system.l3_agents["execution"]
        
        logger.info("DRL L3 Agents instances retrieved.")

        # 2. 设置 DRL 训练环境
        # [已澄清] 架构已确定 (选项 B: 在线)
        # L1/L2 在线运行，作为 DRL 状态的一部分。
        trading_env = TradingEnv(
            data_iterator=data_iterator,
            orchestrator=self.system.orchestrator,
            context_bus=self.system.context_bus
            # (TBD: 奖励函数，状态定义等)
        )
        
        logger.info("TradingEnv initialized.")

        # 3. 设置 DRL 训练器
        drl_trainer = MultiAgentDRLTrainer(
            env=trading_env,
            agents={
                "alpha": l3_alpha_agent,
                "risk": l3_risk_agent,
                "execution": l3_execution_agent
            },
            config=self.config.training.drl_trainer
        )
        
        logger.info("MultiAgentDRLTrainer initialized. Starting DRL backtest/run...")

        # 4. 运行 DRL 循环 (这可能是评估，而不是训练)
        try:
            drl_trainer.run() # 假设 .run() 是评估循环
        except Exception as e:
            logger.critical(f"Unhandled exception during DRL run: {e}", exc_info=True)
        finally:
            logger.info("DRL run finished.")


    def run_live(self):
        """
        Initializes and runs the system in live trading mode.
        [主人喵的修复 11.10] 实现了 TBD
        """
        if not self.system:
            self.setup()
            
        logger.info("Starting Phoenix system in LIVE mode...")
        
        # [主人喵的修复 11.11] 实现了 TBD
        try:
            # 连接到执行网关
            # (假设 execution_gateway 已在 build_system 中初始化并对 order_manager 可用)
            if hasattr(self.system, "order_manager") and hasattr(self.system.order_manager, "connect_live"):
                logger.info("Connecting to execution gateway...")
                self.system.order_manager.connect_live() # [已实现] 假设有这样一个方法
            else:
                logger.warning("OrderManager or connect_live method not found. Skipping gateway connection.")
            
            logger.info("Starting live event distributor...")
            # 假设 self.system.event_distributor.start() 是一个阻塞调用
            # 它会启动消费者 (例如 Kafka) 并开始处理消息
            if not hasattr(self.system, "event_distributor"):
                 logger.error("EventDistributor not found. Cannot run live mode.")
                 return
                 
            self.system.event_distributor.start() 

            # 如果 event_distributor.start() 不是阻塞的，我们可能需要一个主循环
            logger.info("System is live. Waiting for events... (Press Ctrl+C to stop)")
            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("Shutdown signal received. Stopping system...")
            if self.system and hasattr(self.system, 'event_distributor'):
                self.system.event_distributor.stop()
            if self.system and hasattr(self.system, 'order_manager') and hasattr(self.system.order_manager, "disconnect"):
                 self.system.order_manager.disconnect() # 假设
        except Exception as e:
            logger.critical(f"Critical error in live run: {e}", exc_info=True)
        finally:
            logger.info("Phoenix system shut down.")


    def run_training(self):
        """Initializes and runs the training engine."""
        logger.info("Starting Phoenix system in TRAINING mode...")
        # [已澄清] 该模式当前委托给 DRL 训练。
        # (如果需要 L1/L2 训练，需要在此处添加单独的逻辑)
        
        if not self.system:
            self.setup()

        # 1. (TBD) 获取 L1/L2 训练数据
        # data_manager = self.system.data_manager
        # training_data = data_manager.load_training_data("L1_L2_training")

        # 2. (TBD) 初始化 L1/L2 训练引擎
        # training_engine = TrainingEngine(self.config.training, self.system)
        
        # 3. (TBD) 运行 L1/L2 训练
        # training_engine.run(training_data)
        
        logger.warning("Non-DRL training (run_training()) is not implemented. Delegating to DRL...")
        
        # (主人喵的清洁计划 5.3) [新]
        # 让我们假设 'training' 模式是指 DRL 训练
        logger.info("Delegating to DRL Trainer for training...")
        self.run_drl_training()


    # (主人喵的清洁计划 5.3) [新]
    def run_drl_training(self):
        """Runs the DRL training pipeline."""
        if not self.system:
            self.setup()
        
        logger.info("Starting DRL TRAINING run...")
        
        data_iterator = self.system.data_iterator
        if not data_iterator:
            logger.error("Failed to get DataIterator. Aborting DRL training.")
            return

        l3_alpha_agent = self.system.l3_agents["alpha"]
        l3_risk_agent = self.system.l3_agents["risk"]
        l3_execution_agent = self.system.l3_agents["execution"]

        trading_env = TradingEnv(
            data_iterator=data_iterator,
            orchestrator=self.system.orchestrator,
            context_bus=self.system.context_bus,
            # (TBD) 确保 env.reset() 能正确重置 data_iterator
        )
        
        logger.info("TradingEnv initialized for training.")

        drl_trainer = MultiAgentDRLTrainer(
            env=trading_env,
            agents={
                "alpha": l3_alpha_agent,
                "risk": l3_risk_agent,
                "execution": l3_execution_agent
            },
            config=self.config.training.drl_trainer
        )
        
        logger.info("MultiAgentDRLTrainer initialized. Starting training...")

        try:
            drl_trainer.train() # 假设 .train() 是训练循环
        except Exception as e:
            logger.critical(f"Unhandled exception during DRL training: {e}", exc_info=True)
        finally:
            logger.info("DRL training finished.")
            # [已实现] 保存模型
            model_output_path = self.config.get("model_output_path", "models/drl_checkpoints")
            logger.info(f"Saving models to: {model_output_path}")
            drl_trainer.save_models(model_output_path)


@hydra.main(config_path="config", config_name="system", version_base=None)
def main(cfg: DictConfig):
    """Main entry point managed by Hydra."""
    logger.info("Initializing Phoenix Project...")
    
    # [待办] Hydra 会覆盖日志配置。
    # 需要在 config/system.yaml 中配置 Hydra 的日志记录，
    # 以匹配 monitor.logging 的格式。
    
    app = PhoenixProject(cfg)
    
    # [已澄清] 运行模式由配置 (config/system.yaml) 或
    # 命令行覆盖 (例如: python phoenix_project.py run_mode=live) 决定。
    
    mode = cfg.get("run_mode", "backtest")
    
    if mode == "backtest":
        app.run_backtest()
    elif mode == "live":
        app.run_live()
    elif mode == "train":
        # (主人喵的清洁计划 5.3) [修改]
        app.run_drl_training()
    elif mode == "drl_backtest":
        # (主人喵的清洁计划 5.3) [新]
        app.run_drl_backtest()
    else:
        logger.error(f"Unknown run_mode: {mode}")

if __name__ == "__main__":
    main()

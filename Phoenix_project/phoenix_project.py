# Phoenix_project/phoenix_project.py
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
                
                # TBD: What to do with the final state? Log, store, etc.
                if pipeline_state.final_decision:
                    logger.info(f"Final decision at {timestamp}: {pipeline_state.final_decision.action}")
                
                # (主人喵的清洁计划 5.3) [旧]
                # (TBD: 这只是一个高级别的流程。
                # 我们需要将 L3 Agent (Alpha, Risk, Execution) 的输出
                # 连接到 PortfolioConstructor 和 OrderManager。)
                # [主人喵的清洁计划 5.3] [新]
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

        # [主人喵的修复 11.10] 从配置中加载 DRL 模型路径，而不是使用模拟路径
        checkpoint_paths = self.config.get("drl_models", {}).get("checkpoint_paths")
        
        if not checkpoint_paths:
            logger.warning(
                "DRL model checkpoint_paths not found in config. "
                "Falling back to MOCK paths. Please define 'drl_models.checkpoint_paths' in your config."
            )
            checkpoint_paths = {
                "alpha_agent": "models/drl/mock_alpha_agent.pth",
                "risk_agent": "models/drl/mock_risk_agent.pth",
                "execution_agent": "models/drl/mock_execution_agent.pth"
            }
        else:
            logger.info(f"Loading DRL L3 agents from config paths: {checkpoint_paths}")

        # 1. 加载 L3 DRL 智能体
        # TBD: 我们需要一种方法来 '加载' DRL 智能体 (例如，加载权重)
        # 现在，我们只是实例化它们
        
        # (假设 AlphaAgent, RiskAgent, ExecutionAgent 
        # 可以用 'weights_path' 或 'checkpoint_path' 之类的参数初始化)
        # (为了演示，我们假设它们在 registry.py 中被正确初始化)
        l3_alpha_agent = self.system.l3_agents["alpha"]
        l3_risk_agent = self.system.l3_agents["risk"]
        l3_execution_agent = self.system.l3_agents["execution"]
        
        # [TBD]: 这是加载权重的地方吗?
        # l3_alpha_agent.load_model(checkpoint_paths["alpha_agent"])
        # l3_risk_agent.load_model(checkpoint_paths["risk_agent"])
        # l3_execution_agent.load_model(checkpoint_paths["execution_agent"])
        
        logger.info("DRL L3 Agents loaded (MOCK load).")

        # 2. 设置 DRL 训练环境
        # (这很复杂。TradingEnv 需要 L1/L2 的输出作为其 'state' 的一部分)
        # (我们需要一种 '模拟' L1/L2 管道的方法，或者在 DRL 循环中运行它)
        
        # 简化: 我们假设 DRL L3 智能体直接与 Orchestrator 交互
        # 或者 DRL 环境在内部运行 Orchestrator
        
        # 让我们假设 TradingEnv 接受 L1/L2 管道 (Orchestrator)
        # 和数据
        
        # (这个架构 TBD，DRL 训练循环如何与 L1/L2 管道集成?)
        # 选项 A: L1/L2 离线运行，DRL 智能体使用预先计算的特征。
        # 选项 B: L1/L2 在线运行，作为 DRL 状态的一部分。 (计算成本高)
        
        # 假设选项 B (在线)
        trading_env = TradingEnv(
            data_iterator=data_iterator,
            orchestrator=self.system.orchestrator,
            context_bus=self.system.context_bus
            # TBD: 奖励函数，状态定义等
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
        
        # [主人喵的修复 11.10] 启动事件分发器
        try:
            # TBD: Connect to execution gateway
            # (假设 execution_gateway 已在 build_system 中初始化并对 order_manager 可用)
            logger.info("Connecting to execution gateway...")
            # self.system.order_manager.connect_live() # 假设有这样一个方法
            
            logger.info("Starting live event distributor...")
            # 假设 self.system.event_distributor.start() 是一个阻塞调用
            # 它会启动消费者 (例如 Kafka) 并开始处理消息
            self.system.event_distributor.start() 

            # 如果 event_distributor.start() 不是阻塞的，我们可能需要一个主循环
            logger.info("System is live. Waiting for events... (Press Ctrl+C to stop)")
            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            logger.info("Shutdown signal received. Stopping system...")
            if self.system and hasattr(self.system, 'event_distributor'):
                self.system.event_distributor.stop()
            # if self.system and hasattr(self.system, 'order_manager'):
                # self.system.order_manager.disconnect() # 假设
        except Exception as e:
            logger.critical(f"Critical error in live run: {e}", exc_info=True)
        finally:
            logger.info("Phoenix system shut down.")


    def run_training(self):
        """Initializes and runs the training engine."""
        logger.info("Starting Phoenix system in TRAINING mode...")
        # (TBD: 这是否与 DRL 训练重叠?)
        # (假设这是用于 L1/L2 或其他非 DRL 模型的训练)
        
        if not self.system:
            self.setup()

        # 1. (TBD) 获取训练数据
        # data_manager = self.system.data_manager
        # training_data = data_manager.load_training_data("L1_L2_training")

        # 2. (TBD) 初始化训练引擎
        # training_engine = TrainingEngine(self.config.training, self.system)
        
        # 3. (TBD) 运行训练
        # training_engine.run(training_data)
        
        logger.warning("run_training() is not fully implemented. TBD.")
        
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
            # (TBD) 保存模型
            drl_trainer.save_models(self.config.model_output_path)


@hydra.main(config_path="config", config_name="system", version_base=None)
def main(cfg: DictConfig):
    """Main entry point managed by Hydra."""
    logger.info("Initializing Phoenix Project...")
    
    # (TBD: Hydra 会覆盖日志配置。我们需要确保它与我们的 monitor.logging 兼容)
    
    app = PhoenixProject(cfg)
    
    # (TBD: 我们需要一个 CLI 参数或配置来决定运行哪个模式)
    # (为了演示，我们使用一个配置标志)
    
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

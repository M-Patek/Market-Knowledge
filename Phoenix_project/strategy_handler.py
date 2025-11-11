# Phoenix_project/strategy_handler.py
# [主人喵的修复 11.10] 移除了顶部的 "legacy wrapper" 文档字符串。
# [主人喵的实现 11.11] 实现了 TBD 功能，用于 DRL 策略加载和生命周期管理。
# [主人喵的修复 11.12] 实现了参数化策略加载 (TBD-1) 和停用逻辑 (TBD-2)。

import logging
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

class StrategyHandler:
    """
    [已实现]
    This class is responsible for managing the lifecycle of specific trading strategies.
    It is used by the CognitiveEngine or L3 Agents to select, configure,
    or parameterize the strategy logic (e.g., DRL agent selection,
    Alpha/Risk parameter tuning).
    """

    def __init__(self, config: DictConfig, context_bus, l3_agents: dict):
        """
        [已修改] 需要访问 L3 智能体才能加载模型。
        
        Args:
            config: The system configuration (DictConfig).
            context_bus: The main ContextBus instance.
            l3_agents: A dictionary of L3 agent instances 
                       (e.g., {"alpha": AlphaAgent, "risk": RiskAgent, ...})
                       (这些智能体必须有 .load_model(path) 方法)
        """
        self.config = config
        self.context_bus = context_bus
        self.l3_agents = l3_agents
        self.active_strategies = {} # 存储已加载策略的状态
        logger.info("StrategyHandler initialized.")

    def load_strategy(self, strategy_name: str) -> bool:
        """
        [已实现] 加载策略配置和关联的模型（例如 DRL 权重）。
        """
        if strategy_name in self.active_strategies:
            logger.info(f"Strategy '{strategy_name}' is already loaded.")
            return True

        if strategy_name not in self.config.get("strategies", {}):
            logger.error(f"Strategy '{strategy_name}' not found in config.strategies.")
            return False
            
        logger.info(f"Loading strategy: {strategy_name}")
        strategy_config = self.config.strategies[strategy_name]
        
        try:
            # 检查策略类型并执行特定加载逻辑
            strategy_type = strategy_config.get("type")
            
            if strategy_type == "DRL":
                checkpoint_paths = strategy_config.get("checkpoint_paths")
                if not checkpoint_paths:
                    logger.error(f"DRL Strategy '{strategy_name}' has no 'checkpoint_paths' defined.")
                    return False
                
                # [实现] 这解决了 phoenix_project.py 中的 TBD
                logger.info(f"Loading DRL models for strategy: {strategy_name}")
                
                loaded_at_least_one = False
                if "alpha" in self.l3_agents and "alpha" in checkpoint_paths:
                    if hasattr(self.l3_agents["alpha"], "load_model"):
                        self.l3_agents["alpha"].load_model(checkpoint_paths["alpha"])
                        logger.info(f"Loaded AlphaAgent model from: {checkpoint_paths['alpha']}")
                        loaded_at_least_one = True
                    else:
                        logger.warning("AlphaAgent has no 'load_model' method.")
                        
                if "risk" in self.l3_agents and "risk" in checkpoint_paths:
                    if hasattr(self.l3_agents["risk"], "load_model"):
                        self.l3_agents["risk"].load_model(checkpoint_paths["risk"])
                        logger.info(f"Loaded RiskAgent model from: {checkpoint_paths['risk']}")
                        loaded_at_least_one = True
                    else:
                        logger.warning("RiskAgent has no 'load_model' method.")

                if "execution" in self.l3_agents and "execution" in checkpoint_paths:
                    if hasattr(self.l3_agents["execution"], "load_model"):
                        self.l3_agents["execution"].load_model(checkpoint_paths["execution"])
                        logger.info(f"Loaded ExecutionAgent model from: {checkpoint_paths['execution']}")
                        loaded_at_least_one = True
                    else:
                        logger.warning("ExecutionAgent has no 'load_model' method.")
                
                if not loaded_at_least_one:
                    logger.error(f"No models were loaded for DRL strategy '{strategy_name}'. Check config and L3 agents.")
                    return False

            elif strategy_type == "Parametric":
                # [主人喵的修复 11.12] 实现了参数化策略加载 (TBD-1)
                logger.info(f"Loading Parametric strategy: {strategy_name}")
                params = strategy_config.get("params")
                if not params:
                    logger.warning(f"Parametric strategy {strategy_name} has no 'params' defined in config. No parameters applied.")
                    # 即使没有参数，也可能算加载成功，只是激活时无效果
                
                # 假设参数是为 L3 智能体准备的
                # 示例：将参数应用到 L3 智能体
                applied_at_least_one = False
                if "alpha" in self.l3_agents and "alpha" in params:
                    if hasattr(self.l3_agents["alpha"], "set_parameters"):
                        self.l3_agents["alpha"].set_parameters(params["alpha"])
                        logger.info(f"Applied alpha parameters for {strategy_name}.")
                        applied_at_least_one = True
                    else:
                        logger.warning(f"AlphaAgent has no 'set_parameters' method.")
                
                if "risk" in self.l3_agents and "risk" in params:
                    if hasattr(self.l3_agents["risk"], "set_parameters"):
                        self.l3_agents["risk"].set_parameters(params["risk"])
                        logger.info(f"Applied risk parameters for {strategy_name}.")
                        applied_at_least_one = True
                    else:
                        logger.warning(f"RiskAgent has no 'set_parameters' method.")
                
                if not applied_at_least_one and params:
                    logger.warning(f"No parameters were applied for Parametric strategy '{strategy_name}'. Check config and L3 agents.")
            
            else:
                logger.warning(f"Unknown strategy type for '{strategy_name}': {strategy_type}")
                return False # 未知类型，加载失败

            self.active_strategies[strategy_name] = strategy_config
            
            # (TBD: 将策略上下文发布到总线?)
            # 最好在 'activate' 时发布，而不是 'load'
            # self.context_bus.publish("STRATEGY_UPDATE", {"strategy_name": strategy_name, "status": "loaded"})
        
            return True

        except Exception as e:
            logger.error(f"Failed to load strategy '{strategy_name}': {e}", exc_info=True)
            return False

    def activate_strategy(self, strategy_name: str) -> bool:
        """
        [已实现] 激活一个已加载的策略用于实时决策。
        """
        if strategy_name not in self.active_strategies:
            logger.error(f"Strategy '{strategy_name}' must be loaded before activation.")
            return False
            
        logger.info(f"Activating strategy: {strategy_name}")
        
        # [实现] 这会通知 Orchestrator 或 L3 智能体使用这个策略
        # 它们应该订阅 "STRATEGY_CONTEXT" 主题
        try:
            self.context_bus.publish(
                "STRATEGY_CONTEXT", 
                {"active_strategy": strategy_name, "config": self.active_strategies[strategy_name]}
            )
            logger.info(f"Strategy context for '{strategy_name}' published to ContextBus.")
            return True
        except Exception as e:
            logger.error(f"Failed to publish strategy activation to ContextBus: {e}", exc_info=True)
            return False

    def deactivate_strategy(self, strategy_name: str):
        """
        [已实现] 停用一个策略。
        """
        if strategy_name not in self.active_strategies:
            logger.warning(f"Strategy '{strategy_name}' not active or loaded.")
            return
            
        logger.info(f"Deactivating strategy: {strategy_name}")
        
        # [主人喵的修复 11.12] TBD 解决 (TBD-2):
        # 决定在停用时从 active_strategies 中移除（卸载）策略，
        # 以确保状态清晰并释放资源。
        if strategy_name in self.active_strategies:
            del self.active_strategies[strategy_name]
            logger.info(f"Strategy {strategy_name} unloaded from active list.")

        # [实现] 发布一个空上下文，通知系统恢复默认行为
        self.context_bus.publish("STRATEGY_CONTEXT", {"active_strategy": None, "config": {}})
        logger.info(f"Strategy '{strategy_name}' deactivated. Null context published.")

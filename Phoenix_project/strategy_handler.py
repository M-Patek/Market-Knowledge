"""
Strategy Handler
- 加载和管理一个或多个交易策略
- 将 Orchestrator 的事件分派给相关策略
"""
from cognitive.engine import CognitiveEngine
from data_manager import DataManager
from ai.metacognitive_agent import MetacognitiveAgent
from cognitive.portfolio_constructor import PortfolioConstructor
from monitor.logging import get_logger
from typing import Dict, Any

class BaseStrategy:
    """策略的基类"""
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.logger.info(f"Strategy {self.__class__.__name__} initialized.")

    async def on_data(self, data_event):
        """处理传入的市场数据"""
        raise NotImplementedError

    async def run_cognitive_cycle(self):
        """执行策略的认知循环"""
        raise NotImplementedError

class RomanLegionStrategy(BaseStrategy):
    """
    一个具体的策略实现示例。
    """
    
    # 关键修正 (Error 3): 
    # 构造函数现在接受由外部(Orchestrator/StrategyHandler)注入的依赖
    def __init__(
        self, 
        config: Dict[str, Any], 
        data_manager: DataManager,
        metacognitive_agent: MetacognitiveAgent,
        portfolio_constructor: PortfolioConstructor
    ):
        """
        使用依赖注入初始化策略。
        """
        super().__init__(config)
        self.data_manager = data_manager 
        
        # 关键修正: 使用注入的依赖项来构建 CognitiveEngine
        # 而不是错误地传递 data_manager
        self.cognitive_engine = CognitiveEngine(
            config=self.config.get('cognitive_engine', {}),
            metacognitive_agent=metacognitive_agent,
            portfolio_constructor=portfolio_constructor
        )
        
        self.logger.info("RomanLegionStrategy components initialized.")

    async def on_data(self, data_event):
        """处理传入的市场数据"""
        self.logger.debug(f"Received data event: {data_event.get('type')}")
        # (策略的数据处理逻辑)
        pass

    async def run_cognitive_cycle(self):
        """
        执行策略的认知循环并返回信号。
        """
        self.logger.info("Running cognitive cycle...")
        # 1. (获取策略所需的数据)
        # market_data = self.data_manager.get_latest_data(...)
        
        # 2. (运行认知引擎)
        # fusion_result = await self.cognitive_engine.run_cycle(market_data)
        
        # 3. (生成信号)
        # signal = self.portfolio_constructor.generate_signal(fusion_result)
        # return signal
        
        # (模拟返回一个信号)
        from execution.signal_protocol import StrategySignal
        return StrategySignal(
            strategy_id="RomanLegion_v1",
            target_weights={"AAPL": 0.5, "MSFT": 0.5}
        )

class StrategyHandler:
    """
    加载、保存和协调所有活动策略。
    """
    def __init__(self, config, data_manager, metacognitive_agent, portfolio_constructor):
        self.config = config
        self.data_manager = data_manager
        self.metacognitive_agent = metacognitive_agent
        self.portfolio_constructor = portfolio_constructor
        self.strategies: Dict[str, BaseStrategy] = {}
        self.logger = get_logger(self.__class__.__name__)
        self.load_strategies()

    def load_strategies(self):
        """
        根据配置加载策略。
        """
        self.logger.info("Loading strategies...")
        # (此处应有逻辑根据 config 加载策略)
        # 示例:
        strategy_config = self.config.get('strategy_handler', {}).get('strategies', [])
        
        if not strategy_config:
            self.logger.warning("No strategies defined in config. Loading default RomanLegionStrategy.")
            # 修正: 确保在创建时注入所有依赖项
            self.strategies['RomanLegion_v1'] = RomanLegionStrategy(
                config=self.config, # (传递相关配置)
                data_manager=self.data_manager,
                metacognitive_agent=self.metacognitive_agent,
                portfolio_constructor=self.portfolio_constructor
            )

    async def run_all_cognitive_cycles(self):
        """
        触发所有策略的认知循环。
        """
        signals = []
        for name, strategy in self.strategies.items():
            try:
                signal = await strategy.run_cognitive_cycle()
                if signal:
                    signals.append(signal)
            except Exception as e:
                self.logger.error(f"Error running cognitive cycle for strategy {name}: {e}")
        return signals

    async def on_event(self, event):
        """
        将事件分派给所有策略。
        """
        for strategy in self.strategies.values():
            if hasattr(strategy, 'on_event'):
                await strategy.on_event(event)

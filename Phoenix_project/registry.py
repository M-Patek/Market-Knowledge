# registry.py
# (这是一个高级别的示例，展示了依赖注入 (DI) 的设置位置)
# (您的实际文件可能会使用 DI 框架，如 dependency_injector)

from Phoenix_project.config.loader import load_config
from Phoenix_project.core.exceptions import ConfigurationError # (需要导入)

# (导入所有服务和智能体)
from Phoenix_project.data_manager import DataManager
from Phoenix_project.cognitive.engine import CognitiveEngine
from Phoenix_project.cognitive.portfolio_constructor import PortfolioConstructor
from Phoenix_project.execution.order_manager import OrderManager
from Phoenix_project.execution.adapters import SimulatedBrokerAdapter # (或 Alpaca)
from Phoenix_project.execution.trade_lifecycle_manager import TradeLifecycleManager

# [任务 2.2] 导入 L3 智能体和加载器
from Phoenix_project.agents.l3.base import DRLAgentLoader # (假设 L3/base.py 中有此加载器)
from Phoenix_project.agents.l3.alpha_agent import AlphaAgent
from Phoenix_project.agents.l3.risk_agent import RiskAgent
from Phoenix_project.agents.l3.execution_agent import ExecutionAgent

from Phoenix_project.controller.orchestrator import Orchestrator

# (假设... 导入 L1/L2 智能体和子组件...)
# from Phoenix_project.ai.reasoning_ensemble import ReasoningEnsemble
# ...


def build_system(config_path: str = "config/system.yaml"):
    """
    构建并连接整个 Phoenix 系统。
    """
    config = load_config(config_path)

    # --- 构建核心组件 ---
    
    # (构建 L1/L2 认知引擎及其依赖项)
    # ...
    # reasoning_ensemble = ReasoningEnsemble(...)
    # fact_checker = FactChecker(...)
    # ...
    cognitive_engine = CognitiveEngine(
        # ... 传入 L2 依赖项
    )
    
    data_manager = DataManager(config.get('data_manager', {}))
    
    # (构建执行层)
    broker_config = config.get('broker', {})
    # (根据 broker_config 选择 adapter)
    broker_adapter = SimulatedBrokerAdapter() 
    
    trade_manager = TradeLifecycleManager(
        initial_cash=config.get('trading', {}).get('initial_cash', 100000.0)
    )
    
    order_manager = OrderManager(
        broker=broker_adapter,
        trade_lifecycle_manager=trade_manager
        # [TODO]: 也许 ExecutionAgent 应该被注入到 OrderManager 中?
        # execution_agent=...
    )

    # (构建旧的 PortfolioConstructor - 仍然需要，以防万一)
    portfolio_constructor = PortfolioConstructor(
        config=config.get('portfolio', {}),
        risk_manager=None, # (L3 RiskAgent 替换了旧的 RiskManager)
        sizer=None # (L3 AlphaAgent 替换了旧的 Sizer)
    )

    # --- [任务 2.2 配套任务] 实例化 L3 DRL 智能体 ---
    
    l3_config = config.get('l3_agents', {})
    if not l3_config:
        raise ConfigurationError("`l3_agents` config missing in system.yaml. Heart transplant failed.")

    # (我们使用 L3/base.py 中的 DRLAgentLoader 来加载 RLLib 检查点)
    
    alpha_agent = DRLAgentLoader.load_agent(
        AlphaAgent, # 智能体类
        l3_config.get('alpha_agent', {}).get('model_path') # config/system.yaml 中的路径
    )
    risk_agent = DRLAgentLoader.load_agent(
        RiskAgent, 
        l3_config.get('risk_agent', {}).get('model_path')
    )
    execution_agent = DRLAgentLoader.load_agent(
        ExecutionAgent, 
        l3_config.get('execution_agent', {}).get('model_path')
    )
    
    # (检查智能体是否加载成功)
    if not all([alpha_agent, risk_agent, execution_agent]):
        raise ConfigurationError("Failed to load L3 DRL agents from checkpoints. Check paths in system.yaml.")

    # --- 构建 Orchestrator (注入所有依赖项) ---
    
    orchestrator = Orchestrator(
        config=config.get('system', {}),
        data_manager=data_manager,
        cognitive_engine=cognitive_engine,
        portfolio_constructor=portfolio_constructor, # (传入旧的)
        order_manager=order_manager,
        
        # [任务 2.2] 传入新的 L3 智能体
        alpha_agent=alpha_agent,
        risk_agent=risk_agent,
        execution_agent=execution_agent
    )
    
    print("Phoenix System Build Complete. L3 DRL Heart Transplant successful.")
    
    return orchestrator

# (其他辅助 build_... 函数)
# ...
